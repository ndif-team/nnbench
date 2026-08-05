"""One-time fetch: snapshot the MIB IOI task into data/mib/ioi.json.

Reproduces the MIB circuit-track data recipe (hannamw/MIB-circuit-track,
MIB_circuit_track/dataset.py, task='ioi') exactly:

- source: hf.co/datasets/mib-bench/ioi (MIT), `test` split (1000 rows)
- corrupted prompt: the `s2_io_flip_counterfactual` column (the track's default) — the giver
  flips from the subject to the indirect object, so the answer flips between the two names
- filter: subject, indirect object, and random_c must tokenize to the same length (GPT-2
  tokenizer, leading-space form) — this is what makes clean/corrupted BPE-length-matched,
  which the patch cell requires for position-aligned replacement
- answers: (" {indirect_object}", " {subject}") = (correct, incorrect), stored as strings

Rows come from the datasets-server /rows API (public, no auth, no `datasets` dependency).
Run from the repo root with any env that has `transformers`:

    python scripts/fetch_mib_ioi.py
"""
import json
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

DATASET = "mib-bench/ioi"
SPLIT = "test"
COUNTERFACTUAL = "s2_io_flip_counterfactual"
OUT = Path(__file__).resolve().parents[1] / "data" / "mib" / "ioi.json"


def fetch_rows():
    rows, offset = [], 0
    while True:
        url = (f"https://datasets-server.huggingface.co/rows?dataset={DATASET}"
               f"&config=default&split={SPLIT}&offset={offset}&length=100")
        with urllib.request.urlopen(url, timeout=30) as r:
            batch = json.load(r)["rows"]
        rows += [b["row"] for b in batch]
        offset += len(batch)
        if len(batch) < 100:
            return rows


def main():
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")

    def tlen(name):
        return len(tok(f" {name}", add_special_tokens=False).input_ids)

    rows = fetch_rows()
    items, dropped = [], 0
    for row in rows:
        md = row["metadata"]
        if not (tlen(md["indirect_object"]) == tlen(md["subject"]) == tlen(md["random_c"])):
            dropped += 1  # MIB's own name-length filter (dataset.py filter_dataset, task='ioi')
            continue
        items.append({
            "clean": row["prompt"],
            "corrupted": row[COUNTERFACTUAL]["prompt"],
            "answers": [f" {md['indirect_object']}", f" {md['subject']}"],
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"items": items}, indent=1) + "\n")
    print(f"{len(rows)} rows fetched, {dropped} dropped by the name-length filter, "
          f"{len(items)} written to {OUT}")


if __name__ == "__main__":
    main()
