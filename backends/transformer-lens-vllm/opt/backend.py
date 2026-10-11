"""TransformerLens 4.0 with its readout on the GPU: the same bridge and engine as the system's
backend (../backend.py), driving this config's cells.py, registered for `transformer_lens_opt`."""
from ..backend import TransformerLensBackend, create_backend as _create_backend


class TransformerLensOptBackend(TransformerLensBackend):
    name = "transformer_lens_opt"


def create_backend(spec):
    return _create_backend(spec, cls=TransformerLensOptBackend, mode="transformer-lens-opt")
