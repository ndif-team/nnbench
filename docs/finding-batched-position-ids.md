# nnsight batched `trace` silently mispredicts padded prompts (absolute-position models)

**Summary.** A multi-prompt `model.trace([p1, p2, …])` on an **absolute-position** model (GPT-2
family) silently returns wrong per-prompt activations and logits for **every prompt shorter than the
longest one in the batch**. No error is raised. Single-prompt traces, RoPE models (Llama/Mistral/
Qwen), `.generate()`, and the vLLM backend are all unaffected.

## Where it comes from (attribution)

The wrong result is produced by **nnsight's** batched input prep, not a transformers bug:

- nnsight's `LanguageModel` left-pads batches by default (`padding_side="left"`,
  `modeling/language.py:209-210`) and forwards only `input_ids` + `attention_mask`. It **never
  computes `position_ids`** — `grep -rn position_ids src/nnsight/` returns zero hits.
- transformers' GPT-2 `forward`, given `position_ids=None`, defaults to
  `position_ids = arange(0, S)` broadcast across the batch (`modeling_gpt2.py:857-863`); it does
  **not** derive positions from the attention mask.
- Net effect: a left-padded row's real tokens are assigned absolute positions `[k … k+L-1]`
  (`k` = pad count) instead of `[0 … L-1]`, so each real token reads the wrong learned position
  embedding `wpe[i+k]` instead of `wpe[i]`.

transformers behaves exactly as documented — for left-padded batches the *caller* must supply
`position_ids` (HF's `generate()` does so via `prepare_inputs_for_generation`; a bare `forward()`
does not). nnsight is the caller that left-pads **and** calls the bare forward, with the mask in
hand, but omits the `position_ids` correction. So the gap is at the nnsight layer.

## Severity — high, and effectively a step function

Measured on GPT-2-small, 6 prompts, varying the position offset `k` (equivalent to `k` left-pad
tokens; masked pads contribute nothing, so shifting `position_ids` by `k` is exact):

| pad `k` | mean TV(next-token) | top-1 flip rate | final-layer hidden cos |
|--:|--:|--:|--:|
| 0   | 0.000 | 0%   | 1.000 |
| 1   | 0.897 | 83%  | 0.912 |
| 2   | 0.875 | 100% | 0.928 |
| 4   | 0.889 | 100% | 0.815 |
| 8   | 0.912 | 100% | 0.795 |
| 32  | 0.909 | 100% | 0.776 |
| 128 | 0.900 | 100% | 0.509 |

It does **not** scale down gently with pad count. A *single* pad token already moves the next-token
distribution ~0.9 total-variation (out of a max of 1.0) and flips the top-1 for most prompts; **≥2
pad tokens flip 100%**. The effect saturates almost immediately rather than growing with `k`, because
GPT-2's position embeddings are a learned lookup table — adjacent indices are arbitrary, not-close
vectors — so a one-position shift injects a large perturbation at the residual-stream input that the
LayerNorms do not normalize away (final-layer hidden cosine drops to 0.5–0.9: a rotation, not a
rescale).

**Practical consequence:** there is no "safe" mixed-length batch. In any batch, only the longest
prompt(s) (zero padding) are correct; every shorter prompt is corrupted in proportion to nothing —
even one pad token is enough. The corruption is on *all* activations of the padded rows, so it hits
every methodology (logit lens, steering, ablation, patching, caching…), not just final logits.

## What architecture is impacted

The trigger is a specific positional-encoding design, not a particular model:

- **Impacted — learned absolute position embeddings.** A position-indexed lookup table (`wpe`) added
  to the token embedding at the input layer, where the absolute index of each real token feeds the
  result and the forward defaults `position_ids` to a mask-ignoring `arange`. Shifting that index by
  the pad count reads the wrong learned vector. Confirmed on **GPT-2**; the same `wpe`-style design
  covers the **GPT-2 / GPT-Neo** family. (Caveat: some learned-absolute models — e.g. OPT, RoBERTa —
  derive `position_ids` from the mask inside their own forward, so they sidestep it; the precise
  criterion is "does the forward use the raw absolute index, mask-blind," which GPT-2 does.)
- **Not impacted — relative position encodings.** RoPE (Llama, Mistral, Qwen, Gemma, GPT-J,
  GPT-NeoX) and ALiBi (BLOOM, MPT) make attention depend only on the *offset* between tokens, which a
  uniform left-pad shift leaves unchanged — so modern decoder LLMs are unaffected. In practice this
  is the older-model footgun, but GPT-2 is the interpretability workhorse.

## Scope

- **Affected:** mixed-length batches on the impacted architectures above, via the `.trace()` forward
  path. Padded (shorter) rows wrong; the longest (unpadded) row exact.
- **Unaffected:** single-prompt traces (no padding); `.generate()` (HF supplies `position_ids`); and
  the vLLM backend (each prompt runs as its own unpadded request).

## Fix — nnsight PR #673

In nnsight's `_prepare_input` / `_batch`, when left-padding, derive `position_ids` from the mask and
pass them to the forward:

```python
position_ids = attention_mask.long().cumsum(-1) - 1
position_ids.masked_fill_(attention_mask == 0, 0)
```

This corrects the **trace (bare-forward)** path only. `generate()` derives and advances
`position_ids` itself across decode steps, so the correction is stripped before `generate()` (a
static tensor would freeze the padded rows' positions and corrupt their continuation). Verified: with
the correction, the batched trace result is bit-equal to the per-prompt-interactive result on all
prompts. One change fixes every absolute-position model and every methodology at once. Shipped as
nnsight **PR #673** (branch `fix/batched-position-ids`, targets `dev`).

## How the benchmark surfaced it

The interp-workload benchmark un-gated its batched workload after the nnsight async multi-prompt fix
(PR #662). Comparing each backend's batched per-prompt output against a **per-prompt-interactive
reference** (each prompt run alone) showed vLLM batched matching the reference exactly while HF
batched diverged on the padded rows — isolating the cause to nnsight's `position_ids`-free batched
forward rather than any intervention or backend defect.
