"""Real, sizable trace sets per method — the multi-trace benchmark suite (design.md §5).

The point: replace the hand-picked toy sets in `_prompts.py` (PROBE = 8 facts, one CLEAN/CORRUPTED
pair) with reproducible, standard interp task distributions, so each method is oracle-checked over a
real input pool instead of one or two synthesized traces. A benchmark backed by one prompt is a
demo; a benchmark backed by a trace set is a benchmark.

Why a plain prompt pool is enough: the oracle scores the candidate cell against the HF reference
cell (top-1 token agreement + softmax TV) — backend-vs-backend, NOT vs a gold label. So a trace only
needs to yield a confident, non-degenerate distribution; it does not need a "correct" answer. That
lets the pool be any real, diverse text distribution.

Everything here is TEMPLATE-GENERATED and deterministic (seeded) — no data files, no downloads,
byte-stable across runs, and greedy-safe (ordinary text, run argmax). The generators are the
standard interp task families the corpus uses: subject-relation facts (logit lens / steering /
ablation), IOI minimal pairs (activation patching / causal tracing), and k-shot ICL prompts (the
long / many-shot regime that stresses chunked-prefill and prefix caching). Curated corpora
(CounterFact, the ICL suites, multilingual sets) are a drop-in upgrade behind the same signatures.

# ponytail: template banks, not downloaded datasets. Swap a bank for a real corpus when a method
# needs true distribution coverage rather than diverse-but-synthetic surface forms.
"""
from __future__ import annotations

import random

# --- relation banks: (subject, single-token-ish answer) ---------------------------------------
_CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("Germany", "Berlin"), ("Russia", "Moscow"), ("China", "Beijing"), ("Egypt", "Cairo"),
    ("Greece", "Athens"), ("Canada", "Ottawa"), ("Cuba", "Havana"), ("Peru", "Lima"),
    ("Norway", "Oslo"), ("Poland", "Warsaw"), ("Austria", "Vienna"), ("Portugal", "Lisbon"),
    ("Kenya", "Nairobi"), ("Chile", "Santiago"), ("Iran", "Tehran"), ("Sweden", "Stockholm"),
]
_ELEMENTS = [
    ("gold", "Au"), ("oxygen", "O"), ("hydrogen", "H"), ("carbon", "C"), ("iron", "Fe"),
    ("helium", "He"), ("sodium", "Na"), ("silver", "Ag"), ("nitrogen", "N"), ("calcium", "Ca"),
    ("copper", "Cu"), ("zinc", "Zn"), ("lead", "Pb"), ("neon", "Ne"), ("potassium", "K"),
]
_ANTONYMS = [
    ("hot", "cold"), ("up", "down"), ("big", "small"), ("fast", "slow"), ("light", "dark"),
    ("happy", "sad"), ("open", "closed"), ("rich", "poor"), ("young", "old"), ("high", "low"),
    ("left", "right"), ("true", "false"), ("day", "night"), ("love", "hate"), ("win", "lose"),
]

# multiple phrasings per relation -> real surface-form diversity for the same fact
_TEMPLATES = {
    "capital": (_CAPITALS, [
        "The capital of {s} is the city of",
        "{s}'s capital city is",
        "If you visit {s}, its capital is",
    ]),
    "element": (_ELEMENTS, [
        "The chemical symbol for {s} is",
        "In chemistry, {s} is written as the symbol",
        "On the periodic table, {s} has the symbol",
    ]),
    "antonym": (_ANTONYMS, [
        "The opposite of {s} is",
        "In one word, the antonym of {s} is",
        '"{s}" means the opposite of',
    ]),
}

# --- IOI bank: names/places/objects (common English names are single GPT-2 BPE tokens) ---------
_NAMES = ["John", "Mary", "Tom", "Anna", "Paul", "Kate", "Mark", "Lucy", "Alex", "Emma",
          "David", "Sara", "Peter", "Julia", "Sam", "Rose"]
_PLACES = ["store", "park", "school", "office", "garden", "market", "station", "library"]
_OBJECTS = ["drink", "ball", "book", "ring", "note", "key", "card", "coin"]


def factual(n: int = 200, seed: int = 0) -> list[str]:
    """Subject-relation next-token prompts across relation types and phrasings. Drop-in for PROBE:
    a diverse pool of confident single-token completions for the read/last-token methods (logit
    lens, steering, ablation)."""
    rng = random.Random(seed)
    pool = []
    for _key, (bank, phrasings) in _TEMPLATES.items():
        for subj, _ans in bank:
            for tmpl in phrasings:
                pool.append(tmpl.format(s=subj))
    rng.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    # oversample deterministically if asked for more than the base pool
    return [pool[i % len(pool)] for i in range(n)]


def ioi_pairs(n: int = 120, seed: int = 0) -> list[tuple[str, str]]:
    """Length-matched clean/corrupted minimal pairs for activation patching / causal tracing (the
    canonical IOI task). Clean and corrupted differ only in the first subject name (A -> C), so the
    indirect-object answer changes while structure and token count stay identical.

    Returns [(clean, corrupted), ...]. The current activation_patching spec consumes ONE pair; the
    multi-trace form loops the cell over this list (a small cell/driver change, tracked separately).
    """
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        a, b, c = rng.sample(_NAMES, 3)                 # A, B distractor, C corruptor
        place, obj = rng.choice(_PLACES), rng.choice(_OBJECTS)
        stem = "When {x} and %s went to the %s, %s gave a %s to" % (b, place, b, obj)
        clean, corrupt = stem.format(x=a), stem.format(x=c)
        if clean in seen:
            continue
        seen.add(clean)
        out.append((clean, corrupt))
    return out


def few_shot_icl(n: int = 60, k: int = 8, seed: int = 0) -> list[str]:
    """k-shot ICL prompts (antonym task) — the long / many-shot regime. Deliberately long prompts:
    this is the input distribution that pushes a cell into chunked-prefill and prefix-cache
    territory (the serving-feature axis), which the short factual prompts never reach."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        shots = rng.sample(_ANTONYMS, k + 1)
        demos = "\n".join(f"{s} -> {a}" for s, a in shots[:k])
        query = shots[k][0]
        out.append(f"{demos}\n{query} ->")
    return out


# aggregate pool: a single diverse interactive workload spanning short facts + long ICL
def mixed_pool(n_fact: int = 150, n_icl: int = 50, seed: int = 0) -> list[str]:
    return factual(n_fact, seed) + few_shot_icl(n_icl, seed=seed)


def demo() -> None:
    """Self-check: sizes, determinism, IOI length-match, regime separation. No torch, no GPU."""
    f = factual(200)
    assert len(f) == 200 and all(isinstance(p, str) for p in f), "factual size/type"
    assert factual(50) == factual(50), "factual not deterministic"
    assert factual(50, seed=1) != factual(50, seed=0), "seed has no effect"

    pairs = ioi_pairs(120)
    assert len(pairs) == 120, "ioi size"
    assert ioi_pairs(30) == ioi_pairs(30), "ioi not deterministic"
    for clean, corrupt in pairs:
        assert clean != corrupt, "pair not minimal"
        # length-match proxy: identical whitespace-token count (true BPE match needs a tokenizer;
        # names are chosen single-token, so this proxy holds for GPT-2 in practice).
        assert len(clean.split()) == len(corrupt.split()), f"length mismatch: {clean!r}/{corrupt!r}"
        assert clean.split()[1] != corrupt.split()[1], "corruption not at the subject"
        # semantic validity: the giver (before "gave") is a NAME and the object (after "a") is an
        # OBJECT — guards the %-arg-order class of bug (an object can't be the giver).
        toks = clean.split()
        gi = toks.index("gave")
        assert toks[gi - 1] in _NAMES, f"giver is not a name: {clean!r}"
        assert toks[gi + 2] in _OBJECTS, f"object slot is not an object: {clean!r}"
        # the IO answer (the once-appearing name) must be present and single
        subj_a, subj_b = toks[1], toks[3]
        assert subj_a != subj_b and toks[gi - 1] == subj_b, "giver must be the repeated subject B"

    icl = few_shot_icl(60, k=8)
    assert len(icl) == 60, "icl size"
    # regime separation: ICL prompts are materially longer than factual prompts
    assert min(len(p) for p in icl) > max(len(p) for p in f), "icl not longer than factual"

    print(f"OK  factual={len(f)} (unique={len(set(f))})  "
          f"ioi_pairs={len(pairs)}  few_shot_icl={len(icl)}  "
          f"mixed_pool={len(mixed_pool())}")
    print(f"    sample factual : {f[0]!r}")
    print(f"    sample ioi     : clean={pairs[0][0]!r}\n                     corr ={pairs[0][1]!r}")
    print(f"    sample icl     : {icl[0]!r}")


if __name__ == "__main__":
    demo()
