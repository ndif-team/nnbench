"""Release-mode preflight tests (isb/preflight.py); no GPU dependence in the logic checks.

Pins the contract: release mode REFUSES on contamination and names the offender; it never waits;
casual mode never invokes it (pinned at the CLI wiring level by the flag's absence from default
argv construction, checked here via the split passthrough)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.preflight import check_environment, parse_compute_apps  # noqa: E402
from isb.sweep.split import backend_run_commands  # noqa: E402


def test_compute_app_rows_parse_with_process_names():
    csv = ("00000000:41:00.0, 12345, 34741 MiB, /usr/bin/python3\n"
           "00000000:61:00.0, 99, 512 MiB, ray::TrainWorker\n")
    rows = parse_compute_apps(csv)
    assert len(rows) == 2
    assert rows[0]["pid"] == "12345" and rows[0]["used_mb"] == "34741"
    assert rows[1]["name"] == "ray::TrainWorker"        # the finding can NAME the offender


def test_disk_headroom_finding_fires_on_a_full_volume(tmp_path, monkeypatch):
    import isb.preflight as pf

    class FakeUsage:
        free = 1 * 2**30                                # 1 GB free < the 5 GB floor
    monkeypatch.setattr(pf.shutil, "disk_usage", lambda p: FakeUsage)
    # GPU checks may or may not fire on this host; the disk finding must be present regardless
    findings = check_environment(str(tmp_path))
    assert any("GB free" in x for x in findings)


def test_release_flag_reaches_split_children_and_only_when_set():
    runs = backend_run_commands("e.py", "s", ["hf", "vllm_async"], "d", release=True)
    for _, _, argv in runs:
        assert "--release" in argv                      # every run process gates itself
    runs = backend_run_commands("e.py", "s", ["hf", "vllm_async"], "d")
    for _, _, argv in runs:
        assert "--release" not in argv                  # casual mode: the checker never appears


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        if "tmp_path" in fn.__code__.co_varnames:
            print(f"  SKIP {fn.__name__} (pytest fixtures)")
            continue
        fn()
        print(f"  PASS {fn.__name__}")
    print("\ndone.")


if __name__ == "__main__":
    _run_all()
