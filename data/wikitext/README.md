# wikitext (snapshot)

Source: https://huggingface.co/datasets/Salesforce/wikitext (wikitext-103-raw-v1,
CC BY-SA), `test` split, fetched 2026-08 by `scripts/fetch_wikitext.py`: the first
100 body paragraphs (headings and sub-200-character lines skipped), each cropped to
300 characters at a word boundary.

The upstream jacobian lens (anthropics/jacobian-lens) fits its transport maps on
wikitext, so the jacobian-collection workload measures on the same distribution.
Every prompt tokenizes well past position 16, which the collection estimator's
attention-sink skip requires.

Consumed via `isb/data.py` as the `wikitext` prompt source — the default feed for
the jacobian_collect spec.
