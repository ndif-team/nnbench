# MIB circuit-track datasets (snapshot)

Source: https://huggingface.co/datasets/mib-bench/ioi (MIT license), `test` split,
fetched 2026-08 by `scripts/fetch_mib_ioi.py`. MIB: A Mechanistic Interpretability
Benchmark (Mueller et al., 2025); data recipe reproduced from the circuit-track code
(https://github.com/hannamw/MIB-circuit-track, `MIB_circuit_track/dataset.py`, task
`ioi`).

`ioi.json` is `{"items": [{"clean", "corrupted", "answers"}]}`, 1000 items:

- `clean`: the upstream `prompt` (IOI task: "Then, Henry and Phil ... Henry gave a
  basket to", answer = the indirect object, "Phil").
- `corrupted`: the upstream `s2_io_flip_counterfactual` prompt (the circuit track's
  default counterfactual) — the giver flips from the subject to the indirect object,
  so the answer flips between the two names.
- `answers`: `[" {indirect_object}", " {subject}"]` = (correct, incorrect) for the
  clean prompt, leading-space form. The equivalence oracle does not use them (it
  scores backend-vs-backend); they are kept for answer-based metrics (e.g. a
  multi-pair attribution-patching logit difference).

The upstream name-length filter is applied at fetch time (subject, indirect object,
and random name must tokenize to equal length under GPT-2); all 1000 test rows pass,
and every clean/corrupted pair is full-prompt BPE-length-matched — the property the
activation-patching cell requires for position-aligned replacement.

Consumed via `isb/data.py`: `mib/ioi` (clean, corrupted) pairs for activation
patching; `mib/ioi_prompts` (clean prompts only) for ablation.
