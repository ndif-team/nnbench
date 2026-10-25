"""nnsight 0.4.6 (2025-04-24) on HF transformers.

0.4 records the trace body as a graph of proxies and runs it during the forward, so the body cannot
branch on a module output's structure (`isinstance(out, tuple)` sees a proxy). This backend learns
each module's output arity from one plain forward at load, outside any trace, and the cells in
cells_nnsight04.py spell their reads and writes for it under its own interface name. Saved values
are tensors once the trace exits. backend.py selects it when nnsight 0.4 is installed.
"""
from isb.backends.base import Backend, prompts_of


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
        prompts = self.many(prompts, "run")
        with model.trace(prompts):                       # the prompts as one left-padded batch
            saved = build(list(range(len(prompts)))).save()
        return saved.detach().float().cpu()

    def patch(self, model, clean_prompts, corrupted_prompts, capture, patch):
        clean_prompts, corrupted_prompts = self.paired(clean_prompts, corrupted_prompts, "patch")
        with model.trace(clean_prompts):
            ca = capture().save()
        clean_act = ca.detach().float().cpu()
        with model.trace(corrupted_prompts):
            res = patch(clean_act).save()
        return res.detach().float().cpu()

    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        """0.4's per-step form: inside `model.all()` every module access refers to each call in
        turn, so the body runs once per decode step and rows collect in an `nnsight.list()`. The
        body is recorded once as a graph, so `build_step` gets no step index (None)."""
        import nnsight
        import torch

        if not bounded:
            raise ValueError("nnsight 0.4 has no unbounded iteration; its form is model.all()")
        prompts = self.many(prompts, "generate")
        with model.generate(prompts, max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                            do_sample=False, pad_token_id=model.tokenizer.eos_token_id):
            rows = nnsight.list().save()
            with model.all():
                rows.append(build_step(None))
        rows = list(rows)
        if len(rows) != new_tokens:
            raise RuntimeError(f"collected {len(rows)} per-step rows, expected {new_tokens}")
        return torch.stack([r.detach().float().cpu() for r in rows], dim=0)   # [new_tokens, B, ...]

    def plain(self, model, units, *, new_tokens=1):
        """The wrapped transformers module called directly on one left-padded batch, no trace. Ends
        where the cells end: the result copied to the host, so a timer sees the same boundary."""
        import torch

        inputs = model.tokenizer(prompts_of(units), return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        with torch.no_grad():
            if new_tokens == 1:
                out = model._model(**inputs).logits[:, -1, :]
            else:
                out = model._model.generate(**inputs, max_new_tokens=new_tokens,
                                            min_new_tokens=new_tokens, do_sample=False,
                                            pad_token_id=model.tokenizer.eos_token_id)
        return out.cpu()

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
