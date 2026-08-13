# CounterFact (snapshot)

Source: https://rome.baulab.info/data/dsets/counterfact.json (the ROME data release;
Meng et al., "Locating and Editing Factual Associations in GPT", 2022), fetched
2026-08 by `scripts/fetch_counterfact.py`. Snapshot = the first 1000 of 21,919
records in upstream order.

`counterfact.json` is `{"items": [{"prompt", "target_true", "target_new"}]}`:

- `prompt`: the record's rewrite template with the subject filled in ("The mother
  tongue of Danielle Darrieux is") — a subject-relation statement whose next token
  is the fact. The canonical factual-recall prompt distribution for lens reads and
  steering runs.
- `target_true` / `target_new`: the factual and counterfactual next tokens in
  leading-space form (" French" / " English"). The equivalence oracle does not use
  them (it scores backend-vs-backend); they are kept for answer-based metrics
  (e.g. per-item steering toward `target_new`).

Consumed via `isb/data.py` as the `counterfact` prompt source — the default feed
for the logit-lens and steering specs and the family probe workloads
(gen-steering, Qwen, Nemotron).
