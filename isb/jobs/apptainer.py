"""Apptainer launch path, for hosts without Docker (for example NCSA Delta).

The backend config contract is unchanged: `compose.yml` and the `Dockerfile` it builds stay the
single source. This module reads them. A build translates the Dockerfile's ARG/FROM/RUN lines into
an Apptainer definition; a run maps the `runner` service's image, entrypoint, working directory,
environment and volumes onto `apptainer exec --nv`. The launch method is a runner setting, never a
per-backend switch.

Settings (environment):
  ISB_APPTAINER_DIR          .sif images, definitions and named-volume directories
                             (default: <repo>/.apptainer)
  ISB_APPTAINER_BUILD_FLAGS  extra `apptainer build` flags (default: --fakeroot, since RUN steps
                             install packages as root)
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
from pathlib import Path

from .contract import file_digest

_VARIABLE = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def interpolate(value, env):
    """Compose-style `${NAME}` / `${NAME:-default}` substitution (an empty value takes the default)."""
    if isinstance(value, str):
        return _VARIABLE.sub(lambda m: env.get(m.group(1)) or (m.group(2) or ""), value)
    if isinstance(value, list):
        return [interpolate(v, env) for v in value]
    if isinstance(value, dict):
        return {k: interpolate(v, env) for k, v in value.items()}
    return value


def base_dir(root):
    return Path(os.environ.get("ISB_APPTAINER_DIR") or Path(root) / ".apptainer")


def load_service(compose_file, env):
    """The interpolated `runner` service and the top-level named volumes of one compose.yml."""
    import yaml

    config = interpolate(yaml.safe_load(Path(compose_file).read_text()), env)
    services = config.get("services", {})
    if set(services) != {"runner"}:
        raise ValueError(f"{compose_file}: the apptainer launcher runs a single 'runner' service; "
                         f"supporting services ({sorted(set(services) - {'runner'})}) need Docker")
    service = services["runner"]
    if service.get("ports") or service.get("container_name") or service.get("network_mode"):
        raise ValueError(f"{compose_file}: ports, container_name and network_mode are not job-isolated")
    if "image" not in service or "entrypoint" not in service:
        raise ValueError(f"{compose_file}: the runner service needs an image and an entrypoint")
    return service, config.get("volumes") or {}


def sif_path(root, image):
    return base_dir(root) / "images" / (re.sub(r"[^A-Za-z0-9_.-]", "_", image) + ".sif")


def definition(dockerfile, build_args):
    """An Apptainer definition equivalent to a single-stage Dockerfile using ARG, FROM and RUN.
    Any other instruction is refused rather than silently dropped."""
    text = Path(dockerfile).read_text()
    lines, current = [], ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not current and (not stripped or stripped.startswith("#")):
            continue
        current += raw.rstrip() + "\n"
        if not raw.rstrip().endswith("\\"):
            lines.append(current.rstrip("\n"))
            current = ""
    if current:
        raise ValueError(f"{dockerfile}: trailing line continuation")
    values, base, posts = {}, None, []

    def substitute(s):
        return re.sub(r"\$\{(\w+)\}|\$(\w+)",
                      lambda m: values.get(m.group(1) or m.group(2), m.group(0)), s)

    for line in lines:
        instruction, _, rest = line.strip().partition(" ")
        instruction, rest = instruction.upper(), rest.strip()
        if instruction == "ARG":
            name, _, default = rest.partition("=")
            values[name] = build_args.get(name, default)
            if not values[name]:
                raise ValueError(f"{dockerfile}: ARG {name} has no value")
        elif instruction == "FROM":
            if base is not None:
                raise ValueError(f"{dockerfile}: multi-stage builds need Docker")
            base = substitute(rest.split()[0])
        elif instruction == "RUN":
            if base is None:
                raise ValueError(f"{dockerfile}: RUN before FROM")
            posts.append(substitute(rest))
        else:
            raise ValueError(f"{dockerfile}: {instruction} is not translated to Apptainer; "
                             f"extend isb/jobs/apptainer.py or use Docker")
    if base is None:
        raise ValueError(f"{dockerfile}: no FROM")
    body = "\n".join("    " + line for post in posts for line in post.splitlines())
    return f"Bootstrap: docker\nFrom: {base}\n\n%post\n    set -e\n{body}\n"


def build(compose_file, root, env):
    """Build the runner service's image as a .sif from the Dockerfile its compose.yml names."""
    compose_file = Path(compose_file)
    service, _ = load_service(compose_file, env)
    spec = service.get("build")
    if spec is None:
        raise ValueError(f"{compose_file}: no build section; pull the image with apptainer instead")
    if isinstance(spec, str):
        spec = {"context": spec}
    context = (compose_file.parent / spec.get("context", ".")).resolve()
    args = {k: str(v) for k, v in (spec.get("args") or {}).items()}
    target = sif_path(root, service["image"])
    target.parent.mkdir(parents=True, exist_ok=True)
    recipe = target.with_suffix(".def")
    recipe.write_text(definition(context / spec.get("dockerfile", "Dockerfile"), args))
    flags = shlex.split(os.environ.get("ISB_APPTAINER_BUILD_FLAGS", "--fakeroot"))
    partial = target.with_suffix(".sif.partial")
    subprocess.run(["apptainer", "build", "--force", *flags, str(partial), str(recipe)], check=True)
    os.replace(partial, target)
    return target


def command(compose_file, root, env):
    """`apptainer exec` argv and the extra environment for one job. Environment values go in as
    APPTAINERENV_* variables, so values containing commas or quotes pass through unchanged."""
    compose_file = Path(compose_file)
    service, volumes = load_service(compose_file, env)
    image = sif_path(root, service["image"])
    if not image.is_file():
        raise ValueError(f"{image} is missing; run `bench.py build --launcher apptainer` first")
    binds = []
    for entry in service.get("volumes") or []:
        source, destination, *mode = entry.split(":")
        if source in volumes:                               # a named volume -> a host directory
            named = (volumes[source] or {}).get("name") or source
            host = base_dir(root) / "volumes" / named
            host.mkdir(parents=True, exist_ok=True)
        else:
            host = (compose_file.parent / source).resolve()
        binds += ["--bind", ":".join([str(host), destination, *mode])]
    environment = {key: str(value) for key, value in
                   (service.get("environment") or {}).items()}
    environment["CUDA_VISIBLE_DEVICES"] = env["ISB_GPU"]
    argv = ["apptainer", "exec", "--nv", "--cleanenv", "--no-home",
            "--pwd", service.get("working_dir", "/"), *binds, str(image), *service["entrypoint"]]
    return argv, {f"APPTAINERENV_{key}": value for key, value in environment.items()}


def run(argv, extra_env, log, timeout):
    """Run one job in its own process group; on timeout kill the whole group (the engine's
    child processes included) and re-raise."""
    process = subprocess.Popen(argv, env={**os.environ, **extra_env}, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise


def image_identity(image):
    """sha256 of the .sif, cached beside it and keyed by size and modification time."""
    image = Path(image)
    stat = image.stat()
    cache = image.with_suffix(".sha256.json")
    key = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if cache.is_file():
        saved = json.loads(cache.read_text())
        if saved.get("key") == key:
            return saved["sha256"]
    digest = "sha256:" + file_digest(image)
    cache.write_text(json.dumps({"key": key, "sha256": digest}))
    return digest
