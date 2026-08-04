"""One run, one file (design.md §12.10): <name>.pt holds outputs AND provenance.

A run file is self-contained so a plain file move is the whole archival operation: outputs
can never be separated from the stack that produced them. Fresh runs default to the inbox
(`runs/inbox`); archiving is moving the file into a collection directory.
"""
from __future__ import annotations

import os

INBOX = "runs/inbox"


def run_path(out_dir: str, run_name: str) -> str:
    return os.path.join(out_dir, f"{run_name}.pt")


def save_run(out_dir: str, run_name: str, outputs: dict, provenance: dict) -> str:
    import torch

    os.makedirs(out_dir, exist_ok=True)
    path = run_path(out_dir, run_name)
    torch.save({"outputs": outputs, "provenance": provenance}, path)
    return path


def load_run(out_dir: str, run_name: str) -> tuple[dict, dict]:
    """Returns (outputs, provenance)."""
    import torch

    d = torch.load(run_path(out_dir, run_name), map_location="cpu", weights_only=False)
    return d["outputs"], d["provenance"]
