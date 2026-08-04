"""Micro-tier entrypoint — the Level 0/1 primitive map (design.md §3.7, §12.6).

One backend per process (vLLM EngineCore uses spawn; HF and vLLM never share a process here):

    CUDA_VISIBLE_DEVICES=5 conda run -n nnsight-serve-test python scripts/micro.py --backend hf
    CUDA_VISIBLE_DEVICES=5 conda run -n nnsight-serve-test python scripts/micro.py --backend vllm_async

Always run under `timeout`; a HANG verdict aborts the backend's remaining probes.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isb.micro.run import micro_run_file, print_micro_map, run_micro  # noqa: E402
from isb.runfile import INBOX  # noqa: E402


def main():
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["hf", "vllm_async", "vllm_sync"])
    ap.add_argument("--repo", default="openai-community/gpt2")
    ap.add_argument("--probe", nargs="+", default=None, help="run only these probes")
    ap.add_argument("--timeout", type=float, default=180.0, help="per-probe watchdog seconds")
    ap.add_argument("--out", default=INBOX, help="run-file dir; default is the inbox")
    ap.add_argument("--name", default=None, help="run-file name; default <date>_micro_<backend>")
    args = ap.parse_args()

    print(f"[micro] backend={args.backend} repo={args.repo}", flush=True)
    results = run_micro(args.backend, repo=args.repo, only=args.probe, timeout_s=args.timeout)
    print_micro_map(args.backend, args.repo, results)
    name = args.name or f"{time.strftime('%Y-%m-%d')}_constructs_{args.backend}"
    path = micro_run_file(args.backend, args.repo, results, args.out, name)
    print(f"[micro] construct-probe run file -> {path}")


if __name__ == "__main__":
    main()
