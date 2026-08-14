"""One-time fetch: snapshot CounterFact into data/counterfact/counterfact.json.

CounterFact (Meng et al., "Locating and Editing Factual Associations in GPT", the ROME
data release) is the canonical factual-recall prompt distribution: each record is a
subject-relation statement whose next token is the fact ("The mother tongue of Danielle
Darrieux is" -> " French"), plus a counterfactual target (" English").

- source: https://rome.baulab.info/data/dsets/counterfact.json (21,919 records)
- snapshot: the first 1000 records in upstream order (deterministic, no sampling)
- per item: prompt = requested_rewrite.prompt.format(subject); target_true / target_new
  in leading-space form. The equivalence oracle uses only the prompts; the targets are
  kept for answer-based metrics (e.g. per-item steering toward target_new).

Run from the repo root with any python:

    python scripts/fetch_counterfact.py
"""
import json
import urllib.request
from pathlib import Path

URL = "https://rome.baulab.info/data/dsets/counterfact.json"
N = 1000
OUT = Path(__file__).resolve().parents[1] / "data" / "counterfact" / "counterfact.json"


def main():
    with urllib.request.urlopen(URL, timeout=120) as r:
        records = json.load(r)
    items = []
    for rec in records[:N]:
        rw = rec["requested_rewrite"]
        items.append({
            "prompt": rw["prompt"].format(rw["subject"]),
            "target_true": f" {rw['target_true']['str']}",
            "target_new": f" {rw['target_new']['str']}",
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"items": items}, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(records)} records upstream, {len(items)} written to {OUT}")


if __name__ == "__main__":
    main()
