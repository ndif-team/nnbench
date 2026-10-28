"""residual_readout: copy the residual stream after each of the first k layers at each prompt's last
token off the device.

Export methodology, forward only: `[k, B, d_model]` in the engine's activation dtype, the read behind
a probe or monitor on a served request. The rows step k from one layer to every layer, so the cost
of reading more layers and moving them off the device is measured at a fixed forward. Read
methodology -> no effect-size guard. Baseline = the first layer alone, which every row reads.
"""
from ..data import DataRef
from ..sweep.spec import FirstLayers, Regime, Row, Workload

residual_readout = Workload(
    name="residual_readout", methodology="residual_readout",
    rows=tuple(Row(f"{where}, last token", {"layers": layers})
               for where, layers in (
                   ("first layer", FirstLayers(count=1)),
                   ("first seventh of the layers", FirstLayers(1 / 7)),
                   ("first quarter of the layers", FirstLayers(1 / 4)),
                   ("first half of the layers", FirstLayers(1 / 2)),
                   ("first three quarters of the layers", FirstLayers(3 / 4)),
                   ("every layer", FirstLayers(1.0)))),
    # CounterFact factual-recall prompts (data/counterfact/README.md), as for the logit lens
    regimes=(Regime("interactive", DataRef("counterfact", 100)),
             Regime("batched", DataRef("counterfact", 16))),
    baseline={"layers": FirstLayers(count=1)},
)
