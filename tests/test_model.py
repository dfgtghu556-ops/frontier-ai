import math

import pytest
import torch

from frontier_ai.config import ModelConfig
from frontier_ai.model.gpt import GPT, _filter_logits


def tiny_cfg(**kwargs) -> ModelConfig:
    base = dict(vocab_size=32, n_layer=2, n_head=2, n_embd=32, block_size=16)
    base.update(kwargs)
    return ModelConfig(**base)


def test_forward_shapes_and_loss():
    cfg = tiny_cfg()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    out = model(idx, targets=idx)
    assert out.logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert out.loss is not None and torch.isfinite(out.loss)
    # random init: loss should sit near ln(vocab)
    assert 0.5 * math.log(cfg.vocab_size) < out.loss.item() < 2 * math.log(cfg.vocab_size)


def test_causal_masking_left_to_right():
    """Changing tokens *after* position t must not change the logits at t."""
    cfg = tiny_cfg(n_layer=1)
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        logits_a = model(idx).logits
    idx_b = idx.clone()
    idx_b[:, 5:] = torch.randint(0, cfg.vocab_size, (1, cfg.block_size - 5))
    with torch.no_grad():
        logits_b = model(idx_b).logits
    assert torch.allclose(logits_a[:, :5], logits_b[:, :5], atol=1e-5)


@pytest.mark.parametrize("pos", ["learned", "rope"])
@pytest.mark.parametrize("norm", ["rmsnorm", "layernorm"])
@pytest.mark.parametrize("ffn", ["swiglu", "gelu"])
def test_architecture_variants(pos, norm, ffn):
    cfg = tiny_cfg(pos=pos, norm=norm, ffn=ffn)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    out = model(idx, targets=idx)
    assert torch.isfinite(out.loss)


def test_grouped_query_attention():
    cfg = tiny_cfg(n_head=4, n_kv_head=2)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    assert torch.isfinite(model(idx, targets=idx).loss)


def test_generation_matches_full_forward():
    """KV-cached decoding must reproduce a single full-context forward pass."""
    cfg = tiny_cfg(n_layer=2, block_size=16)
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        full = model(idx).logits                      # teacher-forced over all 5 positions
        model(idx[:, :-1])                            # prefill: fills the KV cache
        cache = model._last_cache                     # (k, v) per block
        step = model(idx[:, -1:], kv_cache=cache, start_pos=idx.shape[1] - 1).logits
    assert torch.allclose(full[:, -1], step[:, -1], atol=1e-4)


def test_generate_grows_sequence():
    cfg = tiny_cfg(block_size=32)
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=8, temperature=1.0, top_k=5,
                         generator=torch.Generator().manual_seed(0))
    assert out.shape == (1, 12)
    assert torch.equal(out[:, :4], idx)


def test_generate_never_exceeds_block_size():
    cfg = tiny_cfg(block_size=16)
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 16))  # prompt fills the window
    out = model.generate(idx, max_new_tokens=50, generator=torch.Generator().manual_seed(0))
    assert out.shape[1] == 16  # no room: returns the prompt unchanged

    short = idx[:, :10]
    out = model.generate(short, max_new_tokens=50, generator=torch.Generator().manual_seed(0))
    assert out.shape[1] == 16  # stops at the context limit instead of crashing

    too_long = torch.randint(0, cfg.vocab_size, (1, 40))
    out = model.generate(too_long, max_new_tokens=4, generator=torch.Generator().manual_seed(0))
    assert out.shape[1] == 16  # over-long prompts are truncated, then filled


def test_tied_embeddings_share_storage():
    cfg = tiny_cfg(tie_embeddings=True)
    model = GPT(cfg)
    assert model.lm_head.weight.data_ptr() == model.token_emb.weight.data_ptr()


def test_param_groups_split_decay():
    model = GPT(tiny_cfg())
    groups = model.parameter_groups(weight_decay=0.1)
    total = sum(len(g["params"]) for g in groups)
    assert total == sum(1 for p in model.parameters() if p.requires_grad) > 0
    assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0


def test_top_k_filter_masks_all_but_k():
    logits = torch.arange(10.0).unsqueeze(0).repeat(2, 1)
    filtered = _filter_logits(logits.clone(), top_k=3)
    assert int(torch.isfinite(filtered).sum(dim=-1).unique().item()) == 3


def test_block_size_is_enforced():
    model = GPT(tiny_cfg(block_size=8))
    with pytest.raises(ValueError):
        model(torch.randint(0, 32, (1, 9)))
