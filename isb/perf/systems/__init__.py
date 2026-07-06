"""Per-system perf cells. One module per system, each exposing `run(cfg) -> metrics dict`,
importing only its own deps (vllm-lens, vllm-hook, nnsight never share a process). The runner
(scripts/perf.py) imports exactly one of these per subprocess by `cfg.system`.
"""
