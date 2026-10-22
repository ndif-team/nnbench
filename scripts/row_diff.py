"""Compare two run files row by row: status, recorded error, median latency and output difference.

    python scripts/row_diff.py runs/inbox das-before das-after

The check a cell change runs before it is pushed, on the backends whose code the change touches
(docs/writing-workloads.md §8): a row whose recorded error text changed is a cell-form change
until the commit message says why. Runs come from `scripts/execute.py` (one cell, one engine,
one file) or from a job's exported run files.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from isb.runfile import load_run  # noqa: E402

META = ("__meta__",)
FIELDS = ("status", "median_latency_ms", "median_ms")


def _label(key):
    return "/".join(map(str, key))


def _message(text):
    """The recorded error without what moves between runs of the same code path: the compiled
    trace body's object id (`<edit-140...>`) and traceback line numbers."""
    text = re.sub(r"<edit-\d+>", "<edit>", text or "")
    return re.sub(r"line \d+", "line N", text).strip()


def diff_rows(a, b):
    """Rows of (key, status_a, status_b, error_a, error_b, latency_a, latency_b, max_abs_diff)."""
    meta_a, meta_b = a.get(META, {}), b.get(META, {})
    rows = []
    for key in sorted(set(meta_a) | set(meta_b), key=_label):
        ra, rb = meta_a.get(key, {}), meta_b.get(key, {})
        oa, ob = a.get(key), b.get(key)
        max_diff = None
        if isinstance(oa, torch.Tensor) and isinstance(ob, torch.Tensor):
            max_diff = ((oa.float() - ob.float()).abs().max().item() if oa.shape == ob.shape
                        else f"shapes {list(oa.shape)} vs {list(ob.shape)}")
        rows.append({"row": _label(key),
                     "status": (ra.get("status"), rb.get("status")),
                     "error": (_message(ra.get("error")), _message(rb.get("error"))),
                     "latency_ms": tuple(next((r[f] for f in FIELDS[1:] if f in r), None)
                                         for r in (ra, rb)),
                     "max_abs_diff": max_diff})
    return rows


def main(argv):
    if len(argv) != 4:
        raise SystemExit(__doc__)
    out, name_a, name_b = argv[1:]
    a, _ = load_run(out, name_a)
    b, _ = load_run(out, name_b)
    changed = 0
    for row in diff_rows(a, b):
        same = row["status"][0] == row["status"][1] and row["error"][0] == row["error"][1]
        mark = "  " if same else "!!"
        changed += not same
        shown = {**row, "error": tuple(e[:160] for e in row["error"])}
        print(mark, row["row"], json.dumps({k: v for k, v in shown.items() if k != "row"},
                                           default=str))
    print(f"{changed} row(s) changed status or error text between {name_a} and {name_b}")
    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
