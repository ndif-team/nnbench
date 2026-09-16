"""Reporting policy: numerical uncertainty keeps completed measurements eligible."""

POLICY_VERSION = 1
EXCLUDED_STATES = frozenset({
    "INVALID_OUTPUT", "CONFIRMED_INCORRECT", "ERROR", "HANG", "JOB_FAILED",
    "UNSUPPORTED", "UNSUPPORTED_BY_CONSTRUCTION", "NOT_RUN", "PENDING",
    "RUNNING", "CANCELLED",
})


def performance_eligible(state: str) -> bool:
    """Eligibility on validation grounds; workload/hardware comparability is separate.

    Legacy SILENTLY_WRONG labels alone provide insufficient evidence for exclusion.
    Missing timing is never manufactured by this policy.
    """
    return state not in EXCLUDED_STATES
