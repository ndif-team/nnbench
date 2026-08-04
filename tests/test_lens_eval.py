"""lens_eval task tests (isb/tasks/lens_eval.py) — no GPU; torch only.

Pins the two pieces of task logic that fail silently if broken: the readout-position rules
(input-space, per item) and the upstream pass@k metric (min-over-layers rank of each
intermediate's token). The trace-side read path is the shared _resid + profile machinery,
covered elsewhere.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.tasks.lens_eval import POSITION_RULES, first_token_id, locate, pass_at_k  # noqa: E402


class _FakeTok:
    """Whitespace-free fake: one id per fixed token string, decode inverts encode."""
    def __init__(self, tokens):
        self.tokens = list(tokens)

    def encode(self, text, add_special_tokens=True):
        # the fake's "tokenization" is the fixed token list; text is ignored beyond membership
        return list(range(len(self.tokens)))

    def decode(self, ids):
        return "".join(self.tokens[i] for i in ids)


def test_locate_last_is_final_prompt_token():
    tok = _FakeTok(["Fact", ":", " the", " answer", " is"])
    assert locate(tok, "whatever", "last") == -1


def test_locate_last_newline_finds_the_couplet_break():
    # poetry rule: readout at the LAST newline token (end of line 1 of the couplet)
    tok = _FakeTok(["line", "one", "\n", "line", "two"])
    assert locate(tok, "line one\nline two", "last_newline") == 2
    two_nl = _FakeTok(["a", "\n", "b", "\n", "c"])
    assert locate(two_nl, "a\nb\nc", "last_newline") == 3      # last, not first
    try:
        locate(_FakeTok(["no", "newline"]), "no newline", "last_newline")
        raise AssertionError("expected ValueError on a prompt with no newline token")
    except ValueError:
        pass


def test_locate_rejects_unknown_rule():
    try:
        locate(_FakeTok(["x"]), "x", "before_target")
        raise AssertionError("expected ValueError on an unknown rule")
    except ValueError:
        pass


class _IdTok:
    """first_token_id fake: maps ' word' to a fixed id table."""
    def __init__(self, table):
        self.table = table

    def encode(self, text, add_special_tokens=True):
        return [self.table[text]]


def test_pass_at_k_min_over_layers():
    # 2 layers, vocab 6. intermediate "hit" (id 3): rank 0 at layer 1 -> min rank 0 -> pass@1.
    # intermediate "miss" (id 5): worst logit in both layers -> min rank 5 -> fails even @4.
    tok = _IdTok({" hit": 3, " miss": 5})
    l0 = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])     # id 3 rank 3, id 5 rank 5
    l1 = torch.tensor([1.0, 2.0, 3.0, 9.0, 4.0, 0.0])     # id 3 rank 0, id 5 rank 5
    out = torch.stack([l0, l1])                            # [L=2, vocab] (vllm-shaped, no batch dim)
    items = [{"intermediates": ["hit", "miss"]}]
    assert pass_at_k([out], items, tok, k=1) == 0.5        # hit passes, miss fails
    assert pass_at_k([out], items, tok, k=6) == 1.0        # everything passes at vocab size
    # hf-shaped [L, 1, vocab] must score identically (the reshape normalizes the batch dim)
    assert pass_at_k([out.unsqueeze(1)], items, tok, k=1) == 0.5


def test_position_rules_cover_all_six_shipped_datasets():
    import os
    shipped = sorted(f[:-5] for f in os.listdir(
        Path(__file__).resolve().parents[1] / "data" / "jlens") if f.endswith(".json"))
    assert shipped == sorted(POSITION_RULES)
    assert POSITION_RULES["lens-eval-poetry"] == "last_newline"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
