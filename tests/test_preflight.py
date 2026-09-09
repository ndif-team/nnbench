"""Environment-check logic for the legacy standalone executor; no GPU required."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.preflight import check_environment, parse_compute_apps  # noqa: E402


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
