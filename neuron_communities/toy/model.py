"""The toy model (a 2-layer Llama, so the Build A harvest runs on it unchanged) and final-position MLP interventions.

Interventions act on the MLP post-activation ``h`` (the input of ``down_proj``, the neuron the harvest scores) at the
final position only, like Goodfire's target-token-only interventions. Per layer and per row:

* ``zero``: a (B, d_mlp) 0/1 mask; masked neurons are set to 0;
* ``patch``: a (B, d_mlp) 0/1 mask and a (B, d_mlp) source activation; masked neurons take the source value.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from neuron_communities.toy.tasks import ToyConfig, ToyWorld


def build_model(cfg: ToyConfig, world: ToyWorld) -> LlamaForCausalLM:
    torch.manual_seed(cfg.seed)
    lc = LlamaConfig(vocab_size=world.vocab_size, hidden_size=cfg.d_model, intermediate_size=cfg.d_mlp, num_hidden_layers=2,
                     num_attention_heads=cfg.n_heads, num_key_value_heads=cfg.n_heads, max_position_embeddings=32,
                     bos_token_id=world.bos_tok, eos_token_id=world.eq_tok, tie_word_embeddings=False)
    return LlamaForCausalLM(lc)


@contextmanager
def intervene(model, zero: dict[int, torch.Tensor] | None = None, patch: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
              capture: dict | None = None):
    """Apply final-position interventions inside the block; ``capture`` (a dict) receives each layer's final-position
    ``h`` (B, d_mlp) after the intervention."""
    zero, patch = zero or {}, patch or {}
    handles = []
    for L, layer in enumerate(model.model.layers):
        def hook(_m, ins, L=L):
            h = ins[0]
            last = h[:, -1, :]
            if L in zero:
                last = last * (1 - zero[L].to(last.dtype))
            if L in patch:
                m, src = patch[L]
                m = m.to(last.dtype)
                last = last * (1 - m) + src.to(last.dtype) * m
            if capture is not None:
                capture[L] = last
            if L in zero or L in patch:
                h = torch.cat([h[:, :-1, :], last[:, None, :]], dim=1)
                return (h,)
            return None
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


def final_logits(model, tokens: torch.Tensor, **kw) -> torch.Tensor:
    with intervene(model, **kw):
        out = model.model(input_ids=tokens)
    return model.lm_head(out.last_hidden_state[:, -1])


def capture_h(model, tokens: torch.Tensor) -> dict[int, torch.Tensor]:
    cap: dict = {}
    with torch.no_grad(), intervene(model, capture=cap):
        model.model(input_ids=tokens)
    return {L: v.detach() for L, v in cap.items()}
