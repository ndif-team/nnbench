# Jacobian-lens evaluation datasets (unmodified copy)

Source: https://github.com/anthropics/jacobian-lens (`data/evaluations/`), Apache License
2.0, copied 2026-07 unmodified. `README-upstream.md` is the upstream description of each
set (readout position, scoring). Consumed by `isb/tasks/lens_eval.py`.

Six prompt distributions, each `{"items": [{"prompt", "intermediates", "target"?}]}`:
`intermediates` are the latent/bridge concepts the lens should surface at the readout
position; `target` (where present) only defines that position and is not scored.
