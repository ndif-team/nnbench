"""Construct probes: the Level 0/1 primitive map, one backend per process (vLLM EngineCore uses
spawn; HF and vLLM never share a process here). Run it in an environment with that backend's
stack (transformers for hf, vLLM for vllm_*):

    CUDA_VISIBLE_DEVICES=5 python scripts/constructs.py --backend hf
    CUDA_VISIBLE_DEVICES=5 python scripts/constructs.py --backend vllm_sync

Always run under `timeout`; a HANG verdict aborts the backend's remaining probes.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isb.constructs.run import constructs_run_file, print_constructs_map, run_constructs  # noqa: E402
from isb.runfile import INBOX  # noqa: E402


def main():
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["hf", "vllm_async", "vllm_sync"])
    ap.add_argument("--repo", default="openai-community/gpt2")
    ap.add_argument("--probe", nargs="+", default=None, help="run only these probes")
    ap.add_argument("--timeout", type=float, default=180.0, help="per-probe watchdog seconds")
    ap.add_argument("--out", default=INBOX, help="run-file dir; default is the inbox")
    ap.add_argument("--name", default=None, help="run-file name; default <date>_constructs_<backend>")
    args = ap.parse_args()

    print(f"[constructs] backend={args.backend} repo={args.repo}", flush=True)
    results = run_constructs(args.backend, repo=args.repo, only=args.probe, timeout_s=args.timeout)
    print_constructs_map(args.backend, args.repo, results)
    name = args.name or f"{time.strftime('%Y-%m-%d')}_constructs_{args.backend}"
    path = constructs_run_file(args.backend, args.repo, results, args.out, name)
    print(f"[constructs] run file -> {path}")


if __name__ == "__main__":
    main()
