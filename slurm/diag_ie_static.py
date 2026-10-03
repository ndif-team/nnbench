"""Static vs hooked interp-engine on token-norm generation steering: where do steps diverge?

usage: python3 slurm/diag_ie_static.py OUT.json   (inside the interp-engine image, one GPU)
For each backend and condition (no steer; steer at prefill only; steer at every step) record the
greedy tokens and per-step next-token logits of a few counterfact prompts, then compare."""
import json
import sys

import torch
import torch.nn.functional as F

REPO = "Qwen/Qwen2.5-7B-Instruct"
PROMPTS = ["The Eiffel Tower is located in the city of", "Toko Yasuda, the", "Autonomous University of Madrid, which is located in"]
LAYER, ALPHA, NEW = 14, 6.0, 8


def run(backend):
    from interp_engine import load_model, sync_model, to_address
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(REPO)
    eng = sync_model(load_model(REPO, backend=backend, dtype="bfloat16", gpu_memory_utilization=0.85,
                                extra_vllm_kwargs={"enable_prefix_caching": False}))
    eng.warmup()
    tid = tok(" Rome", add_special_tokens=False)["input_ids"][0]
    d = F.normalize(eng.runner.run(eng.model.unembed_rows([tid]))[0].float(), dim=0)
    point = "resid_post.27"
    out = {}
    for cond, gen, alpha in [("none", True, 0.0), ("prefill", False, ALPHA), ("every", True, ALPHA)]:
        for p in PROMPTS:
            ids = tok(p)["input_ids"]
            lens = None
            if alpha:
                lens = {"specs": [{"op": "steer", "layer": LAYER, "delta": d.tolist(), "strength": alpha,
                                   "max_fraction": 2 * alpha}], "steer_generated": gen,
                        "skip_positions": [], "prompt_len": len(ids)}
            kw = {"lens_intervention": lens} if lens else {}
            res, acts = eng.runner.run(eng.model.capture_generation(ids, [point], max_tokens=NEW,
                                                                    temperature=0.0, **kw))
            rows = acts[to_address(point)]
            logits = eng.decode_residuals(rows[len(ids) - 1:]).float()
            out[(cond, p)] = {"rows": rows.shape[0], "n_prompt": len(ids), "logits": logits.cpu(),
                              "res": repr(res)[:300]}
    eng.shutdown()
    return out


if __name__ == "__main__":
    which = sys.argv[2]
    torch.save(run(which), sys.argv[1])
