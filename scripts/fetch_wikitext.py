"""One-time fetch: snapshot wikitext prompts into data/wikitext/wikitext.json.

The upstream jacobian lens is fitted on Salesforce/wikitext (the published checkpoint's
"Salesforce-wikitext" tag), so the jacobian-collection workload uses the same distribution.
Snapshot recipe, deterministic: the first 100 body paragraphs of the wikitext-103-raw-v1
`test` split (skip headings and short lines), each cropped to 300 characters at a word
boundary. The collection estimator skips the first 16 positions (attention sinks), so every
prompt must tokenize well past that; the 200-character floor guarantees it.

    python scripts/fetch_wikitext.py
"""
import json
import urllib.request
from pathlib import Path

DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-103-raw-v1"
SPLIT = "test"
N = 100
OUT = Path(__file__).resolve().parents[1] / "data" / "wikitext" / "wikitext.json"


def main():
    items, offset = [], 0
    while len(items) < N:
        url = (f"https://datasets-server.huggingface.co/rows?dataset={DATASET}"
               f"&config={CONFIG}&split={SPLIT}&offset={offset}&length=100")
        with urllib.request.urlopen(url, timeout=30) as r:
            batch = json.load(r)["rows"]
        if not batch:
            raise RuntimeError(f"split exhausted at {len(items)} items")
        offset += len(batch)
        for row in batch:
            text = row["row"]["text"].strip()
            if len(text) < 200 or text.startswith("="):    # headings / fragments
                continue
            crop = text[:300].rsplit(" ", 1)[0]            # word-boundary crop
            items.append({"prompt": crop})
            if len(items) == N:
                break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"items": items}, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(items)} prompts written to {OUT}")


if __name__ == "__main__":
    main()
