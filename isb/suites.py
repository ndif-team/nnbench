"""Suites: named run recipes. A suite says which workloads run on which models,
on which backends against which reference, and how much data, which regimes, which realizations and
what timing. `bench.py run --suite <name>` runs one; command-line options override its settings.
Suites never carry intervention semantics: those are the workloads' rows.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass

from .sweep.spec import Concurrent, Offline, Open


@dataclass(frozen=True)
class Entry:
    """Workloads × models, with optional data settings for this entry: `size` resizes each regime's
    default source (batched regimes stay at most 16), `data` ("source[:n]") rebinds every regime,
    `regimes` keeps a subset of regime kinds, `loads` adds load patterns (isb.sweep.spec Offline /
    Concurrent / Open) derived from each workload's own regimes, for the workloads
    that declare the pattern. `split_rows` runs each row as a case of its own, so each row gets
    its own job and engine: a system that builds its engine from the locations a case touches
    (nnsight's `taps`) then serves exactly that row's."""
    workloads: tuple
    models: tuple
    size: int | None = None
    data: str | None = None
    regimes: tuple | None = None
    loads: tuple = ()
    split_rows: bool = False


@dataclass(frozen=True)
class Suite:
    name: str
    entries: tuple
    backends: tuple
    reference: str | None
    realizations: str = "default"       # "default" | "all"
    comparison: str = "correctness"
    timeout: int = 1800


_SMALL = ("logit_lens", "jacobian_lens", "steering", "gen_steering", "activation_patching",
          "gen_patching", "ablation", "attention_pattern", "attribution_patching", "das",
          "jacobian_collect", "activation_harvest", "gen_capture", "residual_readout",
          "steering_layers", "activation_harvest_layers")
_NNSIGHT = ("nnsight-hf/default", "nnsight-vllm/default")
# Plain vLLM at each release a system under comparison runs (vllm-lens and vllm-hook 0.19.1,
# TransformerLens 0.20.2, nnsight 0.8.1 0.27.1, interp-engine 0.28.0), compiled and eager.
_FLOORS = {version: (f"vllm-plain/{version}", f"vllm-plain/{version}/eager")
           for version in ("0-19-1", "0-20-2", "0-27-1", "0-28-0")}

SUITES = {s.name: s for s in (
    # gpt2 + SmolLM2-135M: every workload on one modest GPU, with every realization the nnsight
    # cells declare (the naive ports are the frontier markers). SmolLM2 is the fused-residual family.
    Suite("smoke",
          entries=(Entry(_SMALL, ("gpt2",)), Entry(("logit_lens",), ("smollm2-135m",))),
          backends=_NNSIGHT, reference="nnsight-hf/default", realizations="all"),
    # Qwen2.5-14B for the TP/PP parallelism-equivalence check (GT2): the candidate
    # parallel vLLM config is scored against single-GPU vLLM running the SAME cell, so a cell only
    # needs to run and be deterministic. Multi-GPU settings belong in a backend config; pass it with
    # `--backends nnsight-vllm/default <tp/pp config> --reference nnsight-vllm/default`. Interactive
    # and generation regimes only.
    Suite("parallel",
          entries=(Entry(("logit_lens", "steering", "ablation", "activation_patching", "gen_steering"),
                         ("qwen2.5-14b",), size=32, regimes=("interactive", "generation")),),
          backends=("nnsight-vllm/default",), reference="nnsight-vllm/default",
          realizations="all", comparison="equivalence"),
    # NVIDIA Nemotron 3 Nano, a hybrid Mamba-2 + sparse-attention decoder. The
    # residual stream is additive across all block types, so the residual-stream workloads port
    # unchanged; ablation runs its mixer rows. Measured results for the 4B (2026-06-19) are in
    # findings.md; every vLLM verdict there reduces to the documented gaps
    # (interp-methods-catalog.md). The 30B-A3B MoE needs a multi-GPU config.
    Suite("nemotron",
          entries=(Entry(("logit_lens", "steering", "ablation"), ("nemotron3-nano-4b",), size=32,
                         regimes=("interactive",)),),
          backends=_NNSIGHT, reference="nnsight-hf/default", realizations="all"),
    # Cross-system comparison on Qwen2.5-7B-Instruct (28 layers, the cluster's
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
    # Standard request patterns on nnsight 0.8.1, HF against vLLM, each on the workloads that
    # declare it. Offline is `vllm bench throughput` / MLPerf Offline: 256 units as one sweep, B per
    # call, B=all handing vLLM every unit at once (HF pads them into one batch, which may not fit: a
    # finding); activation patching and steered generation sweep 64 units at B=1 and B=16.
    Suite("nnsight-load-offline",
          entries=(Entry(("steering", "logit_lens", "ablation", "activation_harvest"),
                         ("qwen2.5-7b", "qwen3-32b"), regimes=("offline",),
                         loads=(Offline(256, (1, 16, 64, None)),)),
                   Entry(("activation_patching", "gen_steering"), ("qwen2.5-7b", "qwen3-32b"),
                         regimes=("offline",), loads=(Offline(64, (1, 16)),)),
                   # one prompt per call is ceil(d_model/dim_batch) backward passes (43 at d=4096,
                   # dim_batch 96):
                   # few units, B=1 only (how jacobian-lens shards a corpus)
                   Entry(("jacobian_collect",), ("qwen2.5-7b", "qwen3-32b"),
                         regimes=("offline",), loads=(Offline(8, (1,)),))),
          backends=("nnsight-hf/0-8-1", "nnsight-vllm/0-8-1"),
          reference="nnsight-hf/0-8-1", timeout=7200),
    # Closed-loop serving (`vllm bench serve --max-concurrency`, AIPerf's concurrency sweep): 128
    # single-unit requests from C clients. HF and the sync engine take one request at a time (a
    # serial server); the async engine takes them together.
    Suite("nnsight-load-concurrent",
          entries=(Entry(("steering", "logit_lens", "gen_steering", "gen_capture"),
                         ("qwen2.5-7b", "qwen3-32b"), regimes=("concurrent",),
                         loads=(Concurrent(128, (1, 4, 16, 64)),)),),
          backends=("nnsight-hf/0-8-1", "nnsight-vllm/0-8-1", "nnsight-vllm/0-8-1/async"),
          reference="nnsight-hf/0-8-1", timeout=7200),
    # Open-loop serving (MLPerf Server, `vllm bench serve --request-rate`): 128 single-unit
    # requests arriving at 4, 16 and 64 per second. On Qwen2.5-7B the rates fall below, between and
    # above HF's and the async engine's measured closed-loop capacity (steering 35 and 110 units/s,
    # steered generation about 5 and 58). The sync engine is left out: it serves one request at a
    # time like HF, so its open-loop latency follows from its single-request time.
    Suite("nnsight-load-open",
          entries=(Entry(("steering", "logit_lens", "gen_steering", "gen_capture"),
                         ("qwen2.5-7b", "qwen3-32b"), regimes=("open",),
                         loads=(Open(128, (4, 16, 64)),)),),
          backends=("nnsight-hf/0-8-1", "nnsight-vllm/0-8-1/async"),
          reference="nnsight-hf/0-8-1", timeout=7200),
    Suite("nnsight-versions",
          entries=(Entry(("logit_lens", "steering", "ablation"), ("qwen2.5-7b", "qwen3-32b"),
                         size=32),
                   Entry(("gen_steering", "activation_patching"), ("qwen2.5-7b", "qwen3-32b"),
                         size=16)),
          backends=("nnsight-hf/0-8-1", "nnsight-vllm/0-8-1",
                    "nnsight-hf/0-7", "nnsight-vllm/0-7",
                    "nnsight-hf/0-4-6", "nnsight-hf/0-4-6/era"),
          reference="nnsight-hf/0-8-1", timeout=3600),
    # The same three studies on Qwen2.5-72B-Instruct in bf16, which needs two GPUs: HF splits its
    # layers across them (device_map="auto", the */gpu2 configs), vLLM shards each layer (tensor
    # parallel, the */tp2 configs). The weights come from the group's shared cache
    # (ISB_MODEL_CACHE). 0.7 is left out; the all-layer lens skips B=all, whose saves exceed memory.
    Suite("nnsight-large-versions",
          entries=(Entry(("logit_lens", "steering", "ablation"), ("qwen2.5-72b",), size=32),
                   Entry(("gen_steering", "activation_patching"), ("qwen2.5-72b",), size=16)),
          backends=("nnsight-hf/0-8-1/gpu2", "nnsight-vllm/0-8-1/tp2",
                    "nnsight-hf/0-4-6/gpu2"),
          reference="nnsight-hf/0-8-1/gpu2", timeout=7200),
    Suite("nnsight-large-offline",
          entries=(Entry(("steering", "ablation"), ("qwen2.5-72b",), regimes=("offline",),
                         loads=(Offline(256, (1, 16, 64, None)),)),
                   Entry(("logit_lens",), ("qwen2.5-72b",), regimes=("offline",),
                         loads=(Offline(256, (1, 16, 64)),))),
          backends=("nnsight-hf/0-8-1/gpu2", "nnsight-vllm/0-8-1/tp2",
                    "nnsight-hf/0-4-6/gpu2"),
          reference="nnsight-hf/0-8-1/gpu2", timeout=7200),
    Suite("nnsight-large-concurrent",
          entries=(Entry(("steering", "activation_patching", "gen_steering"), ("qwen2.5-72b",),
                         regimes=("concurrent",), loads=(Concurrent(128, (1, 4, 16, 64)),)),),
          backends=("nnsight-hf/0-8-1/gpu2", "nnsight-vllm/0-8-1/async/tp2",
                    "nnsight-hf/0-4-6/gpu2"),
          reference="nnsight-hf/0-8-1/gpu2", timeout=10800),
    # Llama-3.3-70B-Instruct in bf16 on four 80 GB A100s: HF splits its layers across them
    # (device_map="auto", the */gpu4 configs), vLLM shards each layer (tensor parallel, */tp4).
    # Speed: each workload under the load patterns that match how it is used
    # (docs/request-patterns.md), nnsight 0.8.1 and 0.7 on each engine. The sync engine takes the
    # offline pattern; the async engine, which takes concurrent requests, the serving patterns.
    Suite("llama70b-offline",
          entries=(Entry(("logit_lens",), ("llama3.3-70b",), regimes=("offline",),
                         loads=(Offline(256, (1, 16, 64)),)),
                   Entry(("activation_patching",), ("llama3.3-70b",), regimes=("offline",),
                         loads=(Offline(64, (1,)),))),
          backends=("nnsight-hf/0-8-1/gpu4", "nnsight-hf/0-7/gpu4",
                    "nnsight-vllm/0-8-1/tp4", "nnsight-vllm/0-7/tp4"),
          reference="nnsight-hf/0-8-1/gpu4", timeout=10800),
    Suite("llama70b-concurrent",
          entries=(Entry(("steering",), ("llama3.3-70b",), regimes=("concurrent",),
                         loads=(Concurrent(128, (1, 16, 64)),)),
                   Entry(("gen_steering",), ("llama3.3-70b",), regimes=("concurrent",),
                         loads=(Concurrent(64, (1, 16, 64)),))),
          backends=("nnsight-hf/0-8-1/gpu4", "nnsight-hf/0-7/gpu4",
                    "nnsight-vllm/0-8-1/async/tp4", "nnsight-vllm/0-7/async/tp4"),
          reference="nnsight-hf/0-8-1/gpu4", timeout=10800),
    # Open loop on HF and the async engine (docs/request-patterns.md): 128 steering requests arriving
    # at 4, 16 and 64 per second, below, between and above the two engines' closed-loop capacity.
    Suite("llama70b-open",
          entries=(Entry(("steering",), ("llama3.3-70b",), regimes=("open",),
                         loads=(Open(128, (4, 16, 64)),)),),
          backends=("nnsight-hf/0-8-1/gpu4", "nnsight-hf/0-7/gpu4",
                    "nnsight-vllm/0-8-1/async/tp4", "nnsight-vllm/0-7/async/tp4"),
          reference="nnsight-hf/0-8-1/gpu4", timeout=10800),
    # Coverage: every load pattern the five workloads declare, 16 units each, on nnsight 0.8.1, 0.7
    # and 0.5.15 (2026-01-13). 0.5 has only the sync vLLM engine, so it also serves the serving
    # patterns, one request at a time like HF.
    Suite("llama70b-coverage-offline",
          entries=(Entry(("logit_lens", "steering", "ablation", "activation_patching", "gen_steering"),
                         ("llama3.3-70b",), regimes=("offline",), loads=(Offline(16, (1, 16)),)),),
          backends=("nnsight-hf/0-8-1/gpu4", "nnsight-hf/0-7/gpu4",
                    "nnsight-hf/0-5-15/gpu4", "nnsight-vllm/0-8-1/tp4",
                    "nnsight-vllm/0-7/tp4", "nnsight-vllm/0-5-15/tp4"),
          reference="nnsight-hf/0-8-1/gpu4", timeout=7200),
    Suite("llama70b-coverage-serving",
          entries=(Entry(("logit_lens", "steering", "gen_steering"), ("llama3.3-70b",),
                         regimes=("concurrent", "open"),
                         loads=(Concurrent(16, (4,)), Open(16, (4.0,)))),),
          backends=("nnsight-hf/0-8-1/gpu4", "nnsight-hf/0-7/gpu4",
                    "nnsight-hf/0-5-15/gpu4", "nnsight-vllm/0-8-1/async/tp4",
                    "nnsight-vllm/0-7/async/tp4", "nnsight-vllm/0-5-15/tp4"),
          reference="nnsight-hf/0-8-1/gpu4", timeout=7200),
    # The fastest form each engine offers on nnsight 0.8.1, one prompt per call
    # (docs/large-model-handoff.md on four GPUs): HF splitting the layers (gpu4) or sharding them with
    # transformers tensor parallelism (tp4); on vLLM the default cell, installed edits, and edits with
    # CUDA-graph taps; plain vLLM 0.27.1 compiled for the engine alone.
    Suite("llama70b-fastest",
          entries=(Entry(("logit_lens", "steering", "ablation", "gen_steering"), ("llama3.3-70b",),
                         size=16, regimes=("interactive", "generation")),
                   Entry(("activation_patching",), ("llama3.3-70b",), size=16,
                         regimes=("interactive",))),
          backends=("nnsight-hf/0-8-1/gpu4", "nnsight-hf/0-8-1/tp4",
                    "nnsight-vllm/0-8-1/tp4", "nnsight-vllm/0-8-1/opt/tp4",
                    "nnsight-vllm/0-8-1/opt/taps/tp4", "vllm-plain/0-27-1/tp4"),
          reference="nnsight-hf/0-8-1/gpu4", timeout=3600),
    # Qwen3.5-4B with the upstream fitted Jacobian lens (its `jlens` asset).
    Suite("fitted-jlens",
          entries=(Entry(("jacobian_lens",), ("qwen3.5-4b",)),),
          backends=_NNSIGHT, reference="nnsight-hf/default"),
    # Cost against the number of hooked layers on Qwen2.5-7B, one row per job (each row's engine
    # serves only that row's layers): the steering edit at the first k layers on every system that
    # edits the residual stream, and plain vLLM at each of their releases. interp-engine is left
    # out: its static config bakes taps at every layer whatever the case touches (load_model's
    # `static_points="auto"` default, interp-engine 1.12.0 load.py).
    Suite("layer-count-steering",
          entries=(Entry(("steering_layers",), ("qwen2.5-7b",), split_rows=True),),
          backends=("nnsight-hf/0-8-1", "nnsight-vllm/0-8-1", "nnsight-vllm/0-8-1/opt/taps",
                    "transformer-lens-vllm/opt", "vllm-lens/default",
                    *_FLOORS["0-19-1"], *_FLOORS["0-20-2"], *_FLOORS["0-27-1"]),
          reference="nnsight-hf/0-8-1", timeout=3600),
    # The same layer counts as a read on vLLM-Hook, whose worker captures hidden states only:
    # the residual stream after each of the first k layers at the last token, held in memory
    # (default) or written to disk (disk), against plain vLLM 0.19.1.
    Suite("layer-count-readout",
          entries=(Entry(("residual_readout",), ("qwen2.5-7b",), split_rows=True),),
          backends=("nnsight-hf/0-8-1", "vllm-hook/default", "vllm-hook/disk", *_FLOORS["0-19-1"]),
          reference="nnsight-hf/0-8-1", timeout=3600),
    # Bytes moved from the engine to the client against cost, on every containerized system: the
    # last-token residual stream of the first k layers (B = 1 and 16), the residual stream at every
    # valid token of the first k layers, and the every-layer logit lens. Each harvest call holds up
    # to [n_layers, 1,097 tokens, d_model] on the client, so the harvest keeps its 16 paragraphs.
    Suite("transfer-volume",
          entries=(Entry(("residual_readout", "logit_lens"), ("qwen2.5-7b",), size=32,
                         split_rows=True),
                   Entry(("activation_harvest_layers",), ("qwen2.5-7b",), split_rows=True)),
          backends=("nnsight-hf/0-8-1", "nnsight-vllm/0-8-1", "nnsight-vllm/0-8-1/opt/taps",
                    "vllm-lens/default", "vllm-hook/default", "vllm-hook/disk",
                    "transformer-lens-vllm/opt", "interp-engine-vllm/default",
                    "interp-engine-vllm/static", *(b for floors in _FLOORS.values() for b in floors)),
          reference="nnsight-hf/0-8-1", timeout=3600),
)}


def parse_load(kind: str, text: str):
    """`UNITS:S[,S...]` -> an Offline (S = units per call, or `all`), Concurrent (S = clients) or
    Open (S = arrivals per second, a positive decimal)."""
    units, separator, settings = text.partition(":")
    if not units.isdigit() or int(units) < 1 or not separator or not settings:
        raise ValueError(f"--{kind} takes UNITS:S[,S...] with UNITS positive")
    values = []
    for item in settings.split(","):
        if kind == "offline" and item == "all":
            values.append(None)
        elif kind == "open" and re.fullmatch(r"[0-9]+(\.[0-9]+)?", item) and float(item) > 0:
            values.append(float(item))
        elif kind != "open" and item.isdigit() and int(item) >= 1:
            values.append(int(item))
        else:
            raise ValueError(f"--{kind} setting {item!r} must be a positive "
                             + {"offline": "integer or 'all'", "concurrent": "integer",
                                "open": "decimal"}[kind])
    return {"offline": Offline, "concurrent": Concurrent, "open": Open}[kind](int(units), tuple(values))


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
                skipped += [{"case": spec.name, **row} for row in rows]
                for one in ([dataclasses.replace(spec, tasks=[task]) for task in spec.tasks]
                            if entry.split_rows else [spec]):
                    key = (one.name, tuple(task.label for task in one.tasks))
                    if any(seen_name == one.name and set(labels) & set(key[1])
                           for seen_name, labels in seen):
                        raise ValueError(f"case {one.name} is selected twice")
                    seen.add(key)
                    specs.append(one)
    if not specs:
        raise ValueError("the selection contains no case any model can express")
    return specs, skipped


def suite_cases(name: str):
    """A suite's resolved specs with its own settings (tests and listings)."""
    suite = SUITES[name]
    return cases(suite.entries, suite.realizations)[0]
