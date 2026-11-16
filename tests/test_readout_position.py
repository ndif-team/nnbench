"""Readout-position rules of the lens datasets (isb/data.py), no GPU: the rule each shipped set
uses and the token index each rule picks (input-space, per item)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.data import POSITION_RULES, locate  # noqa: E402


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
