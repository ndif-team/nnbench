"""Invalid effect outputs remain explicit scientific failures with JSON-safe metadata."""
import pytest
import torch

from isb.sweep.guards import compute_effect_size


@pytest.mark.parametrize("value", [None, torch.empty(0, 8), torch.full((1, 8), float("nan")),
                                  torch.full((1, 8), float("inf"))])
@pytest.mark.parametrize("side", ["baseline", "perturbed"])
def test_effect_rejects_missing_empty_and_nonfinite_outputs(value, side):
    valid = torch.ones(1, 8)
    args = (value, valid) if side == "baseline" else (valid, value)
    with pytest.raises(ValueError, match=f"effect {side} must be a nonempty finite tensor"):
        compute_effect_size(*args)


def test_effect_rejects_incompatible_output_shapes():
    with pytest.raises(ValueError, match="comparable shapes"):
        compute_effect_size(torch.ones(2, 8), torch.ones(3, 8))
