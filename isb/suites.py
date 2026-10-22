"""Suites (design.md §12.16): named run recipes. A suite says which workloads run on which models,
on which backends against which reference, and how much data, which regimes, which realizations and
what timing. `bench.py run --suite <name>` runs one; command-line options override its settings.
Suites never carry intervention semantics: those are the workloads' rows.
"""
from __future__ import annotations

from dataclasses import dataclass

from .sweep.spec import Concurrent, Offline


@dataclass(frozen=True)
class Entry:
    """Workloads × models, with optional data settings for this entry: `size` resizes each regime's
    default source (batched regimes stay at most 16), `data` ("source[:n]") rebinds every regime,
    `regimes` keeps a subset of regime kinds, `loads` adds load patterns (isb.sweep.spec Offline /
    Concurrent, design.md §12.17) derived from each workload's own regimes."""
    workloads: tuple
    models: tuple
    size: int | None = None
    data: str | None = None
    regimes: tuple | None = None
    loads: tuple = ()


@dataclass(frozen=True)
class Suite:
    name: str
    entries: tuple
    backends: tuple
    reference: str | None
    realizations: str = "default"       # "default" | "all" (design.md §12.16)
    comparison: str = "correctness"
    timeout: int = 1800


_SMALL = ("logit_lens", "jacobian_lens", "steering", "gen_steering", "activation_patching",
          "gen_patching", "ablation", "attention_pattern", "attribution_patching", "das",
          "jacobian_collect", "activation_harvest", "gen_capture")
_NNSIGHT = ("nnsight-hf/default", "nnsight-vllm/default")

SUITES = {s.name: s for s in (
    # gpt2 + SmolLM2-135M: every workload on one modest GPU, with every realization the nnsight
    # cells declare (the naive ports are the frontier markers). SmolLM2 is the fused-residual family.
    Suite("smoke",
          entries=(Entry(_SMALL, ("gpt2",)), Entry(("logit_lens",), ("smollm2-135m",))),
          backends=_NNSIGHT, reference="nnsight-hf/default", realizations="all"),
    # Qwen2.5-14B for the TP/PP parallelism-equivalence check (GT2, design.md §12.2): the candidate
    # parallel vLLM config is scored against single-GPU vLLM running the SAME cell, so a cell only
    # needs to run and be deterministic. Multi-GPU settings belong in a backend config; pass it with
    # `--backends nnsight-vllm/default <tp/pp config> --reference nnsight-vllm/default`. Interactive
    # and generation regimes only.
    Suite("parallel",
          entries=(Entry(("logit_lens", "steering", "ablation", "activation_patching", "gen_steering"),
                         ("qwen2.5-14b",), size=32, regimes=("interactive", "generation")),),
          backends=("nnsight-vllm/default",), reference="nnsight-vllm/default",
          realizations="all", comparison="equivalence"),
    # NVIDIA Nemotron 3 Nano, a hybrid Mamba-2 + sparse-attention decoder (design.md §12.7). The
    # residual stream is additive across all block types, so the residual-stream workloads port
    # unchanged; ablation runs its mixer rows. Measured results for the 4B (2026-06-19) are in
    # findings.md; every vLLM verdict there reduces to the documented gaps
    # (interp-methods-catalog.md). The 30B-A3B MoE needs a multi-GPU config.
    Suite("nemotron",
          entries=(Entry(("logit_lens", "steering", "ablation"), ("nemotron3-nano-4b",), size=32,
                         regimes=("interactive",)),),
          backends=_NNSIGHT, reference="nnsight-hf/default", realizations="all"),
    # Cross-system comparison (design.md §12.14) on Qwen2.5-7B-Instruct (28 layers, the cluster's
    # model cache has it): every system's documented realization of the same rows, against nnsight
    # on HF.
    Suite("cross-system",
          entries=(Entry(("logit_lens", "steering", "ablation"), ("qwen2.5-7b",), size=32,
                         regimes=("interactive",)),
                   Entry(("gen_steering", "activation_patching", "activation_harvest", "gen_capture"),
                         ("qwen2.5-7b",), size=16)),
          backends=("nnsight-hf/default", "nnsight-vllm/default", "vllm-lens/default",
                    "interp-engine-vllm/default", "transformer-lens-vllm/default",
                    "vllm-plain/0-19-1", "vllm-plain/0-20-2", "vllm-plain/0-28-0"),
          reference="nnsight-hf/default", timeout=3600),
    # nnsight releases on HF and on vLLM, same rows, batch sizes and precision on both sides: 0.7.0
    # and 0.8.1 each on HF against vLLM, and 0.4.6 (2025-04-24) against 0.8.1 on HF. HF runs sdpa
    # attention in bf16; vLLM runs nnsight's documented default, the sync engine, eager. Timings
    # cover the request only: every engine loads once per job, before warm-up.
    # Standard request patterns (design.md §12.17) on nnsight 0.8.1, HF against vLLM. Offline is
    # `vllm bench throughput` / MLPerf Offline: 256 units as one sweep, B per call, B=all handing
    # vLLM every unit at once (HF pads them into one batch, which may not fit: a finding). The
    # async engine is left out: its offline calls would run on a second, sync engine (§12.14).
    Suite("nnsight-load-offline",
          entries=(Entry(("steering", "logit_lens", "ablation"), ("qwen2.5-7b", "qwen3-32b"),
                         regimes=("offline",), loads=(Offline(256, (1, 16, 64, None)),)),),
          backends=("nnsight-hf/sdpa/0-8-1", "nnsight-vllm/sync/0-8-1"),
          reference="nnsight-hf/sdpa/0-8-1", timeout=7200),
    # Closed-loop serving (`vllm bench serve --max-concurrency`, AIPerf's concurrency sweep): 128
    # single-unit requests from C clients. HF and the sync engine take one request at a time (a
    # serial server); the async engine takes them together.
    Suite("nnsight-load-concurrent",
          entries=(Entry(("steering", "activation_patching", "gen_steering"),
                         ("qwen2.5-7b", "qwen3-32b"), regimes=("concurrent",),
                         loads=(Concurrent(128, (1, 4, 16, 64)),)),),
          backends=("nnsight-hf/sdpa/0-8-1", "nnsight-vllm/sync/0-8-1", "nnsight-vllm/0-8-1"),
          reference="nnsight-hf/sdpa/0-8-1", timeout=7200),
    Suite("nnsight-versions",
          entries=(Entry(("logit_lens", "steering", "ablation"), ("qwen2.5-7b", "qwen3-32b"),
                         size=32),
                   Entry(("gen_steering", "activation_patching"), ("qwen2.5-7b", "qwen3-32b"),
                         size=16)),
          backends=("nnsight-hf/sdpa/0-8-1", "nnsight-vllm/sync/0-8-1",
                    "nnsight-hf/sdpa/0-7", "nnsight-vllm/sync/0-7",
                    "nnsight-hf/sdpa/0-4-6", "nnsight-hf/sdpa/0-4-6/era"),
          reference="nnsight-hf/sdpa/0-8-1", timeout=3600),
    # The same three studies on Qwen2.5-72B-Instruct in bf16, which needs two GPUs: HF splits its
    # layers across them (device_map="auto", the */gpu2 configs), vLLM shards each layer (tensor
    # parallel, the */tp2 configs). The weights come from the group's shared cache
    # (ISB_MODEL_CACHE). 0.7 is left out; the all-layer lens skips B=all, whose saves exceed memory.
    Suite("nnsight-large-versions",
          entries=(Entry(("logit_lens", "steering", "ablation"), ("qwen2.5-72b",), size=32),
                   Entry(("gen_steering", "activation_patching"), ("qwen2.5-72b",), size=16)),
          backends=("nnsight-hf/sdpa/0-8-1/gpu2", "nnsight-vllm/sync/0-8-1/tp2",
                    "nnsight-vllm/sync/0-8-1/byref/tp2", "nnsight-hf/sdpa/0-4-6/gpu2"),
          reference="nnsight-hf/sdpa/0-8-1/gpu2", timeout=7200),
    Suite("nnsight-large-offline",
          entries=(Entry(("steering", "ablation"), ("qwen2.5-72b",), regimes=("offline",),
                         loads=(Offline(256, (1, 16, 64, None)),)),
                   Entry(("logit_lens",), ("qwen2.5-72b",), regimes=("offline",),
                         loads=(Offline(256, (1, 16, 64)),))),
          backends=("nnsight-hf/sdpa/0-8-1/gpu2", "nnsight-vllm/sync/0-8-1/tp2",
                    "nnsight-vllm/sync/0-8-1/byref/tp2", "nnsight-hf/sdpa/0-4-6/gpu2"),
          reference="nnsight-hf/sdpa/0-8-1/gpu2", timeout=7200),
    Suite("nnsight-large-concurrent",
          entries=(Entry(("steering", "activation_patching", "gen_steering"), ("qwen2.5-72b",),
                         regimes=("concurrent",), loads=(Concurrent(128, (1, 4, 16, 64)),)),),
          backends=("nnsight-hf/sdpa/0-8-1/gpu2", "nnsight-vllm/0-8-1/tp2",
                    "nnsight-vllm/0-8-1/byref/tp2", "nnsight-hf/sdpa/0-4-6/gpu2"),
          reference="nnsight-hf/sdpa/0-8-1/gpu2", timeout=10800),
    # Qwen3.5-4B with the upstream fitted Jacobian lens (its `jlens` asset).
    Suite("fitted-jlens",
          entries=(Entry(("jacobian_lens",), ("qwen3.5-4b",)),),
          backends=_NNSIGHT, reference="nnsight-hf/default"),
)}


def parse_load(kind: str, text: str):
    """`UNITS:S[,S...]` -> an Offline (S = units per call, or `all`) or Concurrent (S = clients)."""
    from .sweep.spec import Concurrent, Offline

    units, separator, settings = text.partition(":")
    if not units.isdigit() or int(units) < 1 or not separator or not settings:
        raise ValueError(f"--{kind} takes UNITS:S[,S...] with UNITS positive")
    values = []
    for item in settings.split(","):
        if kind == "offline" and item == "all":
            values.append(None)
        elif item.isdigit() and int(item) >= 1:
            values.append(int(item))
        else:
            raise ValueError(f"--{kind} setting {item!r} must be a positive integer"
                             + (" or 'all'" if kind == "offline" else ""))
    if kind == "offline":
        return Offline(int(units), tuple(values))
    return Concurrent(int(units), tuple(values))


def parse_data(text):
    """A DataRef from "source[:n]", or None."""
    from .data import DataRef

    if not text:
        return None
    name, separator, count = text.partition(":")
    if separator and (not count.isdigit() or int(count) < 1):
        raise ValueError("dataset count must be positive")
    return DataRef(name, int(count) if separator else None)


def cases(entries, realizations="default"):
    """Instantiate every (workload, model) of the entries. Returns the resolved specs and the rows
    the models cannot express, as [{"case", "label", "reason"}] ("*" for a whole case)."""
    from .models import MODELS
    from .sweep.spec import NotApplicable, instantiate
    from .workloads import WORKLOADS

    specs, skipped, seen = [], [], set()
    for entry in entries:
        for workload in entry.workloads:
            if workload not in WORKLOADS:
                raise ValueError(f"unknown workload {workload!r}; available: {', '.join(WORKLOADS)}")
            for model in entry.models:
                if model not in MODELS:
                    raise ValueError(f"unknown model {model!r}; available: {', '.join(MODELS)}")
                try:
                    spec, rows = instantiate(WORKLOADS[workload], model, data=parse_data(entry.data),
                                             size=entry.size, regimes=entry.regimes,
                                             realizations=realizations, loads=entry.loads)
                except NotApplicable as reason:
                    skipped.append({"case": f"{workload}/{model}", "label": "*", "reason": str(reason)})
                    continue
                if spec.name in seen:
                    raise ValueError(f"case {spec.name} is selected twice")
                seen.add(spec.name)
                specs.append(spec)
                skipped += [{"case": spec.name, **row} for row in rows]
    if not specs:
        raise ValueError("the selection contains no case any model can express")
    return specs, skipped


def suite_cases(name: str):
    """A suite's resolved specs with its own settings (tests and listings)."""
    suite = SUITES[name]
    return cases(suite.entries, suite.realizations)[0]
