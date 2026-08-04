"""Perf-microbench core: Config, op timing (split gen vs total), row + aggregation.

Builds on isb/perf/timing.py (sync_cuda / peak_mem_mb / force_gc); does not duplicate them.
Cells live in isb/perf/systems/; the runner is scripts/perf.py. See docs/perf-micro-design.md.
"""
from __future__ import annotations

import json
import os
import random
import statistics
from dataclasses import asdict, dataclass
from time import perf_counter

from .timing import force_gc, peak_mem_mb, reset_peak_mem, sync_cuda


def gpu_used_mb():
    """System-wide GPU memory in use (MB) across the CUDA-visible device(s), via NVML.

    NVML uses PHYSICAL indices, so map through CUDA_VISIBLE_DEVICES; sum across the visible set for
    TP. Returns None if NVML is unavailable. This sees the spawned vLLM engine child, which
    torch.cuda.max_memory_allocated does not (the engine runs in a separate process). Read once
    before model load and once after the run; the difference is this cell's footprint. The reading
    is system-wide, so other tenants' allocations on a shared GPU add noise to the delta.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        idxs = [int(x) for x in vis.split(",") if x != ""] if vis else [0]
        used = sum(
            pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).used
            for i in idxs
        )
        pynvml.nvmlShutdown()
        return used / (1024 ** 2)
    except Exception:
        return None

# Config fields copied onto every result row (so aggregation can group/match on them).
ROW_CFG_FIELDS = (
    "system", "op", "repo", "footprint", "token_mode", "phase", "destination",
    "tensor_parallel_size", "prompt_len", "batch", "new_tokens", "enforce_eager", "layer",
)
# Workload key that a cell shares with its pure_vllm baseline (op/footprint/token_mode/
# destination excluded: the baseline is op-agnostic, it is just the engine running the workload).
BASE_KEYS = ("repo", "phase", "prompt_len", "batch", "tensor_parallel_size", "new_tokens")


@dataclass
class Config:
    system: str                       # pure_vllm | nnsight_vllm | vllm_lens | vllm_hook | native_eagle | _echo
    op: str = "none"                  # none | read | steer | qk
    repo: str = "openai-community/gpt2"
    footprint: str = "one"            # one | half | all  (layers the op touches)
    token_mode: str = "all_tokens"    # all_tokens | last_token
    phase: str = "decode"             # prefill | decode
    destination: str = "host"         # host | gpu  (read only)
    tensor_parallel_size: int = 1
    prompt_len: int = 64
    batch: int = 4
    new_tokens: int = 64              # forced to 1 when phase == "prefill"
    layer: int | None = None          # explicit layer for footprint="one" (default: middle)
    # engine + fairness
    enforce_eager: bool = True
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 2048
    # measurement
    n_warmup: int = 5
    n_reps: int = 10
    seed: int = 0
    timeout_s: float = 600.0

    def eff_new_tokens(self) -> int:
        return 1 if self.phase == "prefill" else self.new_tokens

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "Config":
        return cls(**json.loads(s))

    def cell_id(self) -> str:
        m = self.repo.split("/")[-1]
        return (f"{self.system}.{self.op}.{m}.fp-{self.footprint}.{self.token_mode}."
                f"{self.phase}.tp{self.tensor_parallel_size}.b{self.batch}.p{self.prompt_len}."
                f"n{self.eff_new_tokens()}")


def make_prompts(cfg: Config, vocab: int = 10000) -> list[list[int]]:
    """Deterministic synthetic token-id prompts of EXACT length (the data-volume axis needs an
    exact, model-independent length; matches the mini-sglang approach). Systems wrap these into
    their engine's prompt form (TokensPrompt for vLLM, ids for nnsight)."""
    rng = random.Random(cfg.seed)
    return [[rng.randint(5, vocab - 1) for _ in range(cfg.prompt_len)] for _ in range(cfg.batch)]


def layer_indices(cfg: Config, n_layers: int) -> list[int]:
    """The layers an op touches for the footprint axis. `cache` is NOT a separate op; reading many
    layers is just footprint=all of `read` (see design doc)."""
    if cfg.footprint == "all":
        return list(range(n_layers))
    if cfg.footprint == "half":
        return list(range(0, n_layers, 2))
    return [cfg.layer if cfg.layer is not None else n_layers // 2]   # "one"


def time_op(once, *, n_warmup: int, n_reps: int, mem0=None) -> dict:
    """Warm up, then time `n_reps` calls. `once()` returns (output, splits) where `splits` is a
    dict of inner durations in seconds, e.g. {"gen": ..., "retrieve": ...}. The harness measures
    TOTAL latency (sync-bracketed) and collects each split across trials. Returns medians + std +
    peak mem + the last output (so the oracle can check the timed run, never an unverified one).

    Split timer (design fairness control 4): total = the full op incl. retrieval; gen = the inner
    forward-only duration the cell reports. Report both so on-GPU (nnsight, retrieval deferred to
    collect) and copy-out (plugins) are compared fairly.

    `mem0`: NVML system-wide used MB captured by the cell BEFORE model load. When given, peak_mem_mb
    becomes the after-minus-before footprint (sees the spawned engine); otherwise it falls back to
    the in-process torch peak (~0 for an out-of-process engine).
    """
    for _ in range(n_warmup):
        once()
        force_gc()
    reset_peak_mem()
    totals: list[float] = []
    splits: dict[str, list[float]] = {}
    out = None
    for _ in range(n_reps):
        sync_cuda()
        t0 = perf_counter()
        out, sp = once()
        sync_cuda()
        totals.append(perf_counter() - t0)
        for k, v in (sp or {}).items():
            splits.setdefault(k, []).append(v)
        force_gc()
    med = statistics.median
    sd = lambda xs: statistics.pstdev(xs) if len(xs) > 1 else 0.0   # noqa: E731
    res = {
        "median_total_lat_s": med(totals), "std_total_lat_s": sd(totals),
        "peak_mem_mb": peak_mem_mb(),
        "trials_total_s": totals,
    }
    for k, xs in splits.items():
        res[f"median_{k}_lat_s"] = med(xs)
        res[f"std_{k}_lat_s"] = sd(xs)
    # gen latency drives throughput (the in-forward work); fall back to total if a cell reports no split.
    gen = res.get("median_gen_lat_s", res["median_total_lat_s"])
    res["median_gen_lat_s"] = gen
    if mem0 is not None:
        after = gpu_used_mb()
        if after is not None:
            res["gpu_used_after_mb"] = round(after, 1)
            res["peak_mem_mb"] = round(after - mem0, 1)   # this cell's GPU footprint (weights+KV+buffers)
    return res, out


def throughput(cfg: Config, gen_lat_s: float) -> float:
    """Tokens per second over the in-forward (gen) time. Prefill counts prompt tokens; decode counts
    generated tokens (matches the regime being measured)."""
    n = cfg.batch * (cfg.prompt_len if cfg.phase == "prefill" else cfg.eff_new_tokens())
    return n / gen_lat_s if gen_lat_s else 0.0


def row_from(cfg: Config, metrics: dict) -> dict:
    """Assemble a result row: config fields + metrics + derived throughput + cell id."""
    row = {f: getattr(cfg, f) for f in ROW_CFG_FIELDS}
    row["cell"] = cfg.cell_id()
    row.update(metrics)
    g = metrics.get("median_gen_lat_s")
    row["median_tok_per_s"] = throughput(cfg, g) if g else None
    return row


def attach_overhead(rows: list[dict]) -> list[dict]:
    """For each row, overhead vs the pure_vllm baseline sharing its workload key (BASE_KEYS).
    Overhead on throughput: (base - tps)/base, percent. None when no baseline is present."""
    base = {
        tuple(r[k] for k in BASE_KEYS): r["median_tok_per_s"]
        for r in rows
        if r.get("system") == "pure_vllm" and r.get("median_tok_per_s")
    }
    for r in rows:
        b = base.get(tuple(r.get(k) for k in BASE_KEYS))
        tps = r.get("median_tok_per_s")
        r["baseline_tok_per_s"] = b
        r["overhead_pct"] = (None if not b or not tps else round((b - tps) / b * 100, 2))
    return rows
