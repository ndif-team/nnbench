"""The bundled backend tree (backends/README.md): `<system>/<config>`, code from the config's own path."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.jobs import entry, layout  # noqa: E402

BACKENDS = layout.ROOT / "backends"


def test_every_bundled_config_is_system_slash_config_with_code_on_its_path():
    names = layout.discover()
    assert "nnsight-vllm/default" in names and "nnsight-vllm/opt/taps" in names
    for name in names:
        assert len(name.split("/")) >= 2
        backend = layout.nearest(name, "backend.py")
        assert backend is not None, name
        # The resolved code is the config's own directory or one of its ancestors, never a sibling.
        assert layout.config_dir(name).resolve().is_relative_to(backend.parent.resolve()), name


def test_a_config_uses_the_nearest_implementation():
    assert layout.nearest("nnsight-vllm/fp32", "backend.py") == BACKENDS / "nnsight-vllm/backend.py"
    assert layout.nearest("nnsight-vllm/fp32", "cells.py") is None          # nnsight cells: isb/methodologies
    assert layout.nearest("nnsight-vllm/opt/taps", "backend.py") == BACKENDS / "nnsight-vllm/opt/backend.py"
    assert layout.nearest("nnsight-vllm/opt/taps", "cells.py") == BACKENDS / "nnsight-vllm/opt/cells.py"
    assert layout.nearest("vllm-lens/default", "cells.py") == BACKENDS / "vllm-lens/cells.py"
    assert layout.nearest("vllm-lens/opt", "cells.py") == BACKENDS / "vllm-lens/opt/cells.py"
    assert layout.module_name(BACKENDS / "vllm-lens/opt/backend.py") == "backends.vllm-lens.opt.backend"


@pytest.mark.parametrize("name", layout.discover())
def test_compose_reads_only_its_system_directory_and_the_repository(name):
    directory = layout.config_dir(name)
    service = yaml.safe_load((directory / "compose.yml").read_text())["services"]["runner"]
    system = BACKENDS / name.split("/")[0]
    build = service["build"]
    context = build if isinstance(build, str) else build["context"]
    assert (directory / context).resolve() == system.resolve(), name
    assert (system / "Dockerfile").is_file()
    assert service["entrypoint"] == ["python3", "-m", "isb.jobs.entry"]
    sources = [v.split(":")[0] for v in service["volumes"] if not v.startswith(("$", "model-cache"))]
    assert [(directory / s).resolve() for s in sources] == [layout.ROOT.resolve()], name
    assert service["environment"]["ISB_BACKEND"] == f"${{ISB_BACKEND:-{name}}}"


def test_entry_resolves_without_importing_engine_code():
    create_backend, cells = entry.load("transformer-lens-vllm/opt")
    assert callable(create_backend)
    assert cells == BACKENDS / "transformer-lens-vllm/opt/cells.py"
    with pytest.raises(ValueError):
        layout.config_dir("nnsight-vllm")                                   # a system, not a config
    with pytest.raises(ValueError):
        layout.canonical("nnsight-vllm/missing")
    assert layout.canonical("nnsight-vllm") == "nnsight-vllm/default"
