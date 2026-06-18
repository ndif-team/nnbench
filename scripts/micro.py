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

from isb.micro.run import print_micro_map, run_micro  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["hf", "vllm_async", "vllm_sync"])
    ap.add_argument("--repo", default="openai-community/gpt2")
    ap.add_argument("--probe", nargs="+", default=None, help="run only these probes")
    ap.add_argument("--timeout", type=float, default=180.0, help="per-probe watchdog seconds")
    args = ap.parse_args()

    print(f"[micro] backend={args.backend} repo={args.repo}", flush=True)
    results = run_micro(args.backend, repo=args.repo, only=args.probe, timeout_s=args.timeout)
    print_micro_map(args.backend, args.repo, results)


if __name__ == "__main__":
    main()
