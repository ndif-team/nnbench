"""Spec registry — `bench.py --spec <name>` looks up here. One file per methodology holds every
model's spec for it; suites.py groups them into the named suites."""
from .suites import SUITES

SPECS = {s.name: s for suite in SUITES.values() for s in suite}
assert len(SPECS) == sum(map(len, SUITES.values())), "a spec belongs to exactly one suite"


def default_specs():
    """Spec names swept by `bench.py --spec all` — the smoke suite (larger specs run by name)."""
    return [s.name for s in SUITES["smoke"]]


__all__ = ["SPECS", "SUITES", "default_specs"]
