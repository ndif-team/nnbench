"""The bundled backend catalog (backends/README.md): one Dockerfile, compose.yml, configs.yml and
backend.py per system; a backend name is `<system>/<config>`, an entry of that system's catalog."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.jobs import apptainer, entry, layout  # noqa: E402

BACKENDS = layout.ROOT / "backends"
SYSTEMS = {"interp-engine-vllm", "nnsight-hf", "nnsight-vllm", "transformer-lens-vllm", "vllm-hook",
           "vllm-lens", "vllm-plain"}


def test_every_system_holds_the_contract_files_and_no_nested_configs():
    assert set(layout.systems()) == SYSTEMS
    for system in SYSTEMS:
        directory = BACKENDS / system
        files = {p.name for p in directory.iterdir() if p.is_file()}
        assert {"Dockerfile", "compose.yml", "configs.yml", "backend.py"} <= files, system
        nested = [p.name for p in directory.iterdir() if p.is_dir() and p.name != "__pycache__"]
        assert nested == [], system


def test_names_are_catalog_entries_and_a_bare_system_is_its_default():
    names = layout.discover()
    for name in ("nnsight-vllm/default", "nnsight-vllm/async", "nnsight-vllm/0-8-1/opt/taps/tp4",
                 "nnsight-hf/0-8-1/tp2", "vllm-plain/0-19-1/eager", "interp-engine-vllm/static"):
        assert name in names
    assert not any("byref" in name for name in names)
    assert layout.canonical("nnsight-vllm") == "nnsight-vllm/default"
    assert layout.canonical("vllm-plain") == "vllm-plain/default"
    # the default nnsight-vllm realization is the sync engine; async is selected by name
    assert "mode" not in layout.config("nnsight-vllm/default").options
    assert layout.config("nnsight-vllm/0-8-1").options == layout.config("nnsight-vllm/default").options
    assert layout.config("nnsight-vllm/async").options["mode"] == "async"
    assert layout.config("nnsight-vllm/0-8-1/opt/taps").options["taps"] == "auto"
    tp2 = layout.config("nnsight-hf/0-8-1/tp2")
    assert (tp2.gpus, tp2.ranks, tp2.tag) == (2, 2, "isb-nnsight-hf-0-8-1-tf5-19:local")
    assert tp2.build_args["NNSIGHT_REF"] == "de7fec4d207fc8e0deacaa44fb68f8d0412506ef"
    assert layout.config("nnsight-vllm/0-8-1/tp4").options["tensor_parallel_size"] == 4
    for bad in ("nnsight-vllm/missing", "missing", "nnsight-vllm/../nnsight-hf/default", ""):
        with pytest.raises(ValueError):
            layout.canonical(bad)


def test_each_image_is_built_one_way_from_its_dockerfile():
    for system in SYSTEMS:
        args_of = {}
        dockerfile = (BACKENDS / system / "Dockerfile").read_text()
        for chosen in layout.catalog(system).values():
            assert args_of.setdefault(chosen.tag, chosen.build_args) == chosen.build_args, chosen.name
            for arg in chosen.build_args:
                assert f"ARG {arg}" in dockerfile, (chosen.name, arg)
            recipe = apptainer.definition(chosen.directory / "Dockerfile", chosen.build_args)
            assert recipe.startswith("Bootstrap: docker\nFrom: "), chosen.name
    assert layout.config("nnsight-vllm/0-8-1").tag == layout.config("nnsight-vllm/0-8-1/async/tp2").tag
    assert "From: vllm/vllm-openai:v0.27.1" in apptainer.definition(
        BACKENDS / "nnsight-vllm/Dockerfile", layout.config("nnsight-vllm/0-8-1").build_args)


@pytest.mark.parametrize("system", sorted(SYSTEMS))
def test_compose_is_parameterized_by_the_launcher_environment(system):
    env = {"ISB_IMAGE": "probe:local", "ISB_BACKEND": f"{system}/x", "ISB_ENGINE_OPTIONS": '{"a":1}',
           "ISB_GPUS": "4,5", "ISB_RANKS": "2", "ISB_JOB_DIR": "/j", "ISB_OUTPUT_DIR": "/o"}
    service, volumes = apptainer.load_service(BACKENDS / system / "compose.yml", env)
    assert service["image"] == "probe:local" and "build" not in service
    assert service["entrypoint"] == ["python3", "-m", "isb.jobs.entry"]
    workspace = [v.split(":")[0] for v in service["volumes"] if v.split(":")[1] == "/workspace"]
    assert [Path(s) for s in workspace] == [layout.ROOT.resolve()]
    assert "/j:/job:ro" in service["volumes"] and "/o:/output" in service["volumes"]
    environment = service["environment"]
    assert environment["ISB_BACKEND"] == f"{system}/x" and environment["ISB_ENGINE_OPTIONS"] == '{"a":1}'
    assert environment["ISB_RANKS"] == "2"
    assert apptainer.gpus(service, env) == ["4,5"]
    assert "model-cache" in volumes


def test_entry_resolves_the_system_code_without_importing_engines():
    create_backend, cells = entry.load("transformer-lens-vllm/opt")
    assert callable(create_backend) and cells == BACKENDS / "transformer-lens-vllm/cells.py"
    _, cells = entry.load("nnsight-vllm/0-8-1/opt")
    assert cells is None            # nnsight's cells register through isb.methodologies and the mode
    with pytest.raises(ValueError):
        entry.load("")
    assert layout.module_name(BACKENDS / "vllm-lens/cells_opt.py") == "backends.vllm-lens.cells_opt"


@pytest.mark.parametrize("text,error", [
    ("images: {a: {}}\nconfigs: {default: {image: b}}\n", "names image"),
    ("images: {a: {}}\nconfigs: {default: {image: a, extra: 1}}\n", "takes"),
    ("images: {a: {}}\nconfigs: {default: {image: a, options: {}, gpus: 0}}\n", "positive integer"),
    ("images: {a: {}}\nconfigs: {default: {image: a}}\n", "options mapping"),
    ("images: {a: {}}\nconfigs: {'bad name': {image: a}}\n", "config name"),
    ("images: {a: [x]}\nconfigs: {default: {image: a}}\n", "ARG names"),
    ("images: {a: {}}\n", "mappings"),
])
def test_catalog_rejects_malformed_entries(tmp_path, text, error):
    system = tmp_path / "backends" / "probe"
    system.mkdir(parents=True)
    (system / "configs.yml").write_text(text)
    with pytest.raises(ValueError, match=error):
        layout.catalog("probe", tmp_path)
