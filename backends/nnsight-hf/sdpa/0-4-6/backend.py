"""nnsight 0.4.6 (2025-04-24, the release current in May 2025) on HF transformers.

0.4 builds an intervention graph of proxies at trace time and runs it during the forward, so the
trace body cannot branch on a module output's structure (`isinstance(out, tuple)` sees a proxy).
This backend learns each module's output arity from one plain forward at load, outside any trace,
and the cells (cells.py) spell their reads and writes for it. Its own interface name keeps it off
the 0.7/0.8 cells in isb/methodologies/. Saved values are tensors once the trace exits (0.4.x).
"""
from isb.backends.base import Backend
from isb.jobs.worker import merge_requirements, options
from isb.runs import EngineConfig, RunConfig


class NNsight04Backend(Backend):
    name = "hf_nnsight_0_4"

    def __init__(self, dtype: str | None = None, attn_implementation: str = "eager",
                 device_map: str | None = None):
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.device_map = device_map            # "auto": split across the visible GPUs
        self._arity = {}

    def load(self, repo: str, device: str = "cuda:0"):
        import torch
        import transformers
        from nnsight import LanguageModel

        kw = {}
        if self.dtype is not None:   # transformers 5 renamed torch_dtype to dtype
            key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
            kw[key] = getattr(torch, self.dtype)
        model = LanguageModel(repo, device_map=self.device_map or device, dispatch=True,
                              attn_implementation=self.attn_implementation, **kw)
        self._arity = {}
        return model

    def arity(self, model, envoys):
        """{envoy path: tuple length, or 0 for a bare tensor} for each envoy's module output, from
        one plain forward with temporary hooks; cached per path for the backend's lifetime."""
        import torch

        missing = [e for e in envoys if e.path not in self._arity]
        if missing:
            seen, handles = {}, []
            for envoy in missing:
                def hook(module, args, output, path=envoy.path):
                    seen.setdefault(path, len(output) if isinstance(output, tuple) else 0)
                handles.append(envoy._module.register_forward_hook(hook))
            try:
                inputs = model.tokenizer("a b", return_tensors="pt").to(model.device)
                with torch.no_grad():
                    model._model(**inputs)
            finally:
                for handle in handles:
                    handle.remove()
            self._arity.update(seen)
        return [self._arity[e.path] for e in envoys]

    def run(self, model, prompts, build):
        with model.trace(prompts):          # a list of prompts -> a left-padded batch
            saved = build().save()
        return saved.detach().float().cpu()

    def patch(self, model, clean_prompt, corrupted_prompt, capture, patch):
        with model.trace(clean_prompt):
            ca = capture().save()
        clean_act = ca.detach().float().cpu()
        with model.trace(corrupted_prompt):
            res = patch(clean_act).save()
        return res.detach().float().cpu()

    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        """0.4's per-step idiom: inside `model.all()` every module access refers to each call in
        turn, so the body runs once per decode step; rows collect in an `nnsight.list()`."""
        import nnsight
        import torch

        if not bounded:
            raise ValueError("nnsight 0.4 has no unbounded iteration; its idiom is model.all()")
        prompt = prompts[0] if isinstance(prompts, (list, tuple)) else prompts
        with model.generate(prompt, max_new_tokens=new_tokens, do_sample=False,
                            pad_token_id=model.tokenizer.eos_token_id):
            rows = nnsight.list().save()
            with model.all():
                rows.append(build_step())
        rows = list(rows)
        if len(rows) != new_tokens:
            raise RuntimeError(f"collected {len(rows)} per-step rows, expected {new_tokens}")
        return torch.cat([r.detach().float().cpu() for r in rows], dim=0)

    def vanilla(self, model, prompt, *, new_tokens):
        """The wrapped transformers module called directly, with no trace."""
        import torch

        inputs = model.tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            if new_tokens == 1:
                model._model(**inputs)
            else:
                model._model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False,
                                      pad_token_id=model.tokenizer.eos_token_id)

    def vanilla_many(self, model, prompts, *, new_tokens):
        """`vanilla` over several prompts as one left-padded batch (the offline denominator)."""
        import torch

        inputs = model.tokenizer(list(prompts), return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            if new_tokens == 1:
                model._model(**inputs)
            else:
                model._model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False,
                                      pad_token_id=model.tokenizer.eos_token_id)

    def last(self, t):
        return t[:, -1, :]

    def teardown(self, model) -> None:
        import gc

        import torch

        try:
            del model
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def create_backend(spec):
    params = merge_requirements(spec.hf_kwargs, options())
    if params.pop("force_text_causal", False):
        raise ValueError("the nnsight 0.4 backend does not load multimodal-wrapper checkpoints")
    return NNsight04Backend(**params), RunConfig(EngineConfig("transformers", params=params))
