"""Applicability states (design.md §8.1, §12.2)."""


class AppState:
    SUPPORTED = "SUPPORTED"
    SUPPORTED_DEGRADED = "SUPPORTED_DEGRADED"
    ERROR = "ERROR"
    SILENTLY_WRONG = "SILENTLY_WRONG"      # ran, no error, but != reference (the dangerous cell)
    HANG = "HANG"
    UNSUPPORTED = "UNSUPPORTED"            # no cell implemented for this combination
    NO_REFERENCE = "NO_REFERENCE"          # ran, but the per-family HF control failed -> can't judge
