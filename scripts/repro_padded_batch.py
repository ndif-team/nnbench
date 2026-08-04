"""Padded-batch vs per-prompt on GPT-2 (HF): measure the left-pad absolute-position artifact.

Modes:
  raw     - plain transformers forward, tokenizer left-pad, no position_ids
  nnsight - nnsight LanguageModel trace: batched list-prompt trace vs per-prompt traces

Select the nnsight branch by interpreter, not PYTHONPATH alone (pp-on-dev needs its compiled
nnsight._c extension, so a bare source-tree PYTHONPATH fails to import):
  dev:       nnsight-tf python with PYTHONPATH=/disk/u/zikai/nnsight/src
  pp-on-dev: nnsight-vllm python (editable install points at the pp-on-dev worktree)
The script prints the resolved nnsight path; findings.md cites the 2026-07-24 numbers.
"""
import os, sys
import torch

PROMPTS = [
    "The Eiffel Tower is located in the city of",
    "Rome",
    "The quick brown fox jumps over the lazy dog and then the quick brown fox jumps over the",
    "Water boils at a temperature of",
]

def report(tag, batched, per):
    # batched, per: [N, vocab] final-position logits
    for i in range(len(PROMPTS)):
        d = (batched[i] - per[i]).abs().max().item()
        tv = 0.5 * (batched[i].softmax(-1) - per[i].softmax(-1)).abs().sum().item()
        t1 = (batched[i].argmax() == per[i].argmax()).item()
        print(f"{tag} row{i} len={len(PROMPTS[i].split()):2d} maxabs={d:.4g} tv={tv:.4g} top1_match={t1}")

def run_raw():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    print("transformers", transformers.__version__)
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", torch_dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        enc = tok(PROMPTS, return_tensors="pt", padding=True).to("cuda")
        batched = model(**enc).logits[:, -1].cpu()
        per = []
        for p in PROMPTS:
            e = tok(p, return_tensors="pt").to("cuda")
            per.append(model(**e).logits[0, -1].cpu())
    report("raw", batched, per)

def run_nnsight():
    import nnsight
    print("nnsight from", os.path.dirname(nnsight.__file__))
    from nnsight import LanguageModel
    model = LanguageModel("openai-community/gpt2", device_map="cuda", torch_dtype=torch.float32, dispatch=True)
    per = []
    for p in PROMPTS:
        with model.trace(p):
            out = model.output.logits[0, -1].save()
        per.append(out.detach().cpu())
    with model.trace(PROMPTS):
        outb = model.output.logits[:, -1].save()
    report("nnsight", outb.detach().cpu(), per)

if __name__ == "__main__":
    torch.manual_seed(0)
    {"raw": run_raw, "nnsight": run_nnsight}[sys.argv[1]]()
