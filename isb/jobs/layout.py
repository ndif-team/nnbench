"""The backend catalog: `backends/<system>/configs.yml` (backends/README.md).

A system directory holds one `Dockerfile`, one `compose.yml`, one `configs.yml` and the system's
code (`backend.py`, `cells.py`). `configs.yml` lists the system's `images` (Dockerfile build
arguments, keyed by image name) and its `configs`, each naming its image, its engine `options` and
the `gpus` and `ranks` it reserves. A backend name is `<system>/<config>`; a bare system name means
`<system>/default`. Shared by the host launcher and the in-container entrypoint.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEGMENT = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
DEFAULT = "default"
CONFIG_KEYS = {"image", "options", "gpus", "ranks"}


@dataclass(frozen=True)
class Config:
    """One runnable configuration of a system."""
    name: str            # <system>/<config>
    system: str
    directory: Path      # the system directory: Dockerfile, compose.yml, backend.py, cells.py
    image: str           # image key in the system's configs.yml
    build_args: dict     # Dockerfile ARG values of that image
    options: dict        # the engine options the worker receives (ISB_ENGINE_OPTIONS)
    gpus: int = 1        # host GPUs the job reserves
    ranks: int = 1       # processes per job (torchrun ranks when above 1)

    @property
    def tag(self):
        return f"isb-{self.system}-{self.image}:local"


def split(name):
    """(system, config) of a backend name; a bare system name means its default config."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"backend name {name!r} must be <system>/<config>")
    system, _, config = name.partition("/")
    config = config or DEFAULT
    if not all(SEGMENT.fullmatch(p) and p not in {".", ".."} for p in [system, *config.split("/")]):
        raise ValueError(f"backend name {name!r} must be <system>/<config>")
    return system, config


def systems(root=ROOT):
    """Every system directory under backends/ that holds a configs.yml."""
    base = Path(root) / "backends"
    return sorted(p.name for p in base.iterdir()
                  if SEGMENT.fullmatch(p.name) and (p / "configs.yml").is_file())


def catalog(system, root=ROOT):
    """{config key: Config} of one system, validated from its configs.yml."""
    import yaml

    directory = Path(root) / "backends" / system
    path = directory / "configs.yml"
    if not SEGMENT.fullmatch(system) or not path.is_file():
        raise ValueError(f"unknown system {system!r}; available: {', '.join(systems(root))}")
    data = yaml.safe_load(path.read_text()) or {}
    images, configs = data.get("images"), data.get("configs")
    if not isinstance(images, dict) or not isinstance(configs, dict) or not configs:
        raise ValueError(f"{path}: needs `images` and `configs` mappings")
    for key, args in images.items():
        if not SEGMENT.fullmatch(str(key)) or not isinstance(args or {}, dict) or not all(
                isinstance(k, str) and isinstance(v, (str, int, float)) and not isinstance(v, bool)
                for k, v in (args or {}).items()):
            raise ValueError(f"{path}: image {key!r} must map Dockerfile ARG names to values")
    found = {}
    for key, entry in configs.items():
        key = str(key)
        try:
            split(f"{system}/{key}")
        except ValueError:
            raise ValueError(f"{path}: config name {key!r} must be <name>[/<name>...]") from None
        entry = entry or {}
        if not isinstance(entry, dict) or set(entry) - CONFIG_KEYS:
            raise ValueError(f"{path}: config {key!r} takes {sorted(CONFIG_KEYS)} only")
        image = str(entry.get("image"))
        if image not in images:
            raise ValueError(f"{path}: config {key!r} names image {image!r}, "
                             f"not one of {sorted(map(str, images))}")
        options = entry.get("options")
        if not isinstance(options, dict):
            raise ValueError(f"{path}: config {key!r} needs an options mapping "
                             "({} for the engine's defaults)")
        gpus, ranks = entry.get("gpus", 1), entry.get("ranks", 1)
        for label, value in (("gpus", gpus), ("ranks", ranks)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{path}: config {key!r}: {label} must be a positive integer")
        found[key] = Config(f"{system}/{key}", system, directory, image,
                            {k: str(v) for k, v in (images[image] or {}).items()}, options, gpus, ranks)
    return found


def discover(root=ROOT):
    """Every backend name, `<system>/<config>`."""
    return sorted(c.name for s in systems(root) for c in catalog(s, root).values())


def config(name, root=ROOT):
    """The Config a backend name selects; a bare system name selects `<system>/default`."""
    system, key = split(name)
    found = catalog(system, root)
    if key not in found:
        raise ValueError(f"unknown backend {name!r}; available: {', '.join(discover(root))}")
    return found[key]


def canonical(name, root=ROOT):
    """The full `<system>/<config>` name."""
    return config(name, root).name


def system_dir(name, root=ROOT):
    """The system directory of a backend name (no catalog read: the container entrypoint's path)."""
    return Path(root) / "backends" / split(name)[0]


def module_name(path, root=ROOT):
    """Import name of a file under backends/ (directory names may contain '-', which importlib
    accepts; code inside imports its siblings relatively)."""
    relative = Path(path).relative_to(Path(root)).with_suffix("")
    return ".".join(relative.parts)
