"""A compact decoder-only transformer ("nano-GPT" style, with modern defaults).

Design goals
------------
* **Readable** - the whole model is ~250 lines and has no hidden magic.
* **Modern** - RMSNorm, SwiGLU FFN, optional RoPE, grouped-query attention, no
  biases, tied embeddings, `scaled_dot_product_attention` (flash/mem-efficient
  kernels when available).
* **Device-agnostic** - no CUDA-only code paths; the same module trains on CPU.
* **Fast to sample from** - `generate()` uses a KV cache.

Notes on sizing (defaults are the tiny CPU config):
    n_layer=4, n_head=4, n_embd=128, block_size=128  -> ~0.8M params
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


# --------------------------------------------------------------------- norms --
class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (LLaMA-style)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "rmsnorm":
        return RMSNorm(dim)
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"unknown norm kind '{kind}'")


# --------------------------------------------------------------------- rope --
class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE), applied to the head dimension.

    Cached cos/sin table grows on demand, so generation past `block_size` works.
    """

    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 8192) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, start: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        if start + seq_len > self.cos_cached.shape[2]:
            self._build_cache(max(int(1.5 * (start + seq_len)), self.max_seq_len))
        return (
            self.cos_cached[:, :, start : start + seq_len, :],
            self.sin_cached[:, :, start : start + seq_len, :],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


# ------------------------------------------------------------------- blocks --
class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with grouped-query KV heads and a KV cache."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if rope is not None:
            cos, sin = rope
            # cached RoPE tables are indexed at the absolute position of each token
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        if kv_cache is not None:
            k = torch.cat((kv_cache[0], k), dim=2)
            v = torch.cat((kv_cache[1], v), dim=2)
        new_cache = (k, v)

        if self.n_rep > 1:  # expand grouped KV heads to match query heads
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # flash / mem-efficient kernels when the shapes allow, math kernel otherwise
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=kv_cache is None,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.resid_dropout(self.out_proj(y)), new_cache


class MLP(nn.Module):
    """SwiGLU (default) or GELU feed-forward network."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.ffn_hidden
        if cfg.ffn == "swiglu":
            self.w_in = nn.Linear(cfg.n_embd, 2 * hidden, bias=cfg.bias)
            self.w_out = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        else:
            self.w_in = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
            self.w_out = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.act = nn.SiLU() if cfg.ffn == "swiglu" else nn.GELU()
        self.kind = cfg.ffn
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "swiglu":
            a, b = self.w_in(x).chunk(2, dim=-1)
            return self.dropout(self.w_out(self.act(a) * b))
        return self.dropout(self.w_out(self.act(self.w_in(x))))


class Block(nn.Module):
    """Pre-norm transformer block (norm before attention/MLP, residual add)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = make_norm(cfg.norm, cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = make_norm(cfg.norm, cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        h, cache = self.attn(self.norm1(x), rope=rope, kv_cache=kv_cache)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, cache


# -------------------------------------------------------------------- model --
@dataclass
class ForwardOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class GPT(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False  # toggled by TrainConfig.grad_checkpointing
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if cfg.pos == "learned":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        else:
            self.pos_emb = None  # type: ignore[assignment]
        self.rope = RotaryEmbedding(cfg.head_dim) if cfg.pos == "rope" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = make_norm(cfg.norm, cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        # scaled residual init (GPT-2 style) for the projection layers
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("w_out.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    # ------------------------------------------------------------- init ----
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------- io ------
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def n_params(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def config_dict(self) -> dict:
        return self.cfg.__dict__.copy()

    # --------------------------------------------------------- forward ----
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        start_pos: int = 0,
    ) -> ForwardOutput:
        """Forward pass.

        idx      : (B, T) token ids
        targets  : (B, T) next-token targets; when given, the loss is returned too
        kv_cache : per-block (k, v) tuples for incremental decoding
        """
        B, T = idx.shape
        if T + start_pos > self.cfg.block_size:
            raise ValueError(
                f"sequence position {T + start_pos} exceeds block_size {self.cfg.block_size}"
            )

        x = self.token_emb(idx)
        if self.pos_emb is not None:
            positions = torch.arange(start_pos, start_pos + T, device=idx.device)
            x = x + self.pos_emb(positions)[None, :, :]
        x = self.drop(x)

        rope = None
        if self.rope is not None:
            rope = self.rope(T + start_pos, start=start_pos)

        new_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training and kv_cache is None:
                # checkpoint() needs at least one input with requires_grad
                x, cache = torch.utils.checkpoint.checkpoint(  # type: ignore[assignment]
                    block, x, rope, kv_cache[i] if kv_cache else None, use_reentrant=False
                )
            else:
                x, cache = block(x, rope=rope, kv_cache=kv_cache[i] if kv_cache else None)
            new_cache.append(cache)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1)
            )
        self._last_cache = new_cache  # consumed by generate()
        return ForwardOutput(logits=logits, loss=loss)

    # ----------------------------------------------------- optimization ----
    def parameter_groups(self, weight_decay: float, decay_matrices: bool = True) -> list[dict]:
        """Split params into decay / no-decay groups (standard GPT recipe)."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_matrix = p.ndim >= 2
            if decay_matrices and is_matrix and not name.endswith("norm_f.weight"):
                decay.append(p)
            else:
                no_decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def estimate_flops(self, batch: int, seq: int) -> float:
        """Rough forward+backward FLOPs per token-batch (6 * N * tokens heuristic)."""
        return 6.0 * self.n_params() * batch * seq

    # ------------------------------------------------------- generation ----
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
        stop_at_eos: int | None = None,
    ) -> torch.Tensor:
        """Sample continuations using a KV cache.

        idx may be shorter than block_size (it is truncated if longer); generation
        stops at the context limit, since learned/RoPE positions cannot extrapolate.
        """
        self.eval()
        out = idx
        # never feed more than block_size tokens, and leave room for the new tokens
        if out.shape[1] > self.cfg.block_size:
            out = out[:, -self.cfg.block_size :]
        max_new_tokens = min(max_new_tokens, self.cfg.block_size - out.shape[1])
        if max_new_tokens <= 0:  # prompt already fills the context window
            return out

        kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
        start_pos = 0
        for _ in range(max_new_tokens):
            ctx = out if kv_cache is None else out[:, -1:]
            logits_out = self.forward(ctx, kv_cache=kv_cache, start_pos=start_pos)
            kv_cache = self._last_cache
            start_pos += ctx.shape[1]
            logits = logits_out.logits[:, -1, :] / max(temperature, 1e-6)
            logits = _filter_logits(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            out = torch.cat((out, next_id), dim=1)
            if stop_at_eos is not None and bool((next_id == stop_at_eos).all()):
                break
        return out


def _filter_logits(
    logits: torch.Tensor, top_k: int | None = None, top_p: float = 1.0
) -> torch.Tensor:
    """Top-k and/or nucleus (top-p) filtering. Returns logits with -inf elsewhere."""
    if top_k is not None and 0 < top_k < logits.size(-1):
        kth = torch.topk(logits, top_k).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum - F.softmax(sorted_logits, dim=-1) > top_p
        remove[:, 0] = False
        logits = logits.masked_fill(
            torch.zeros_like(logits, dtype=torch.bool).scatter_(1, sorted_idx, remove), float("-inf")
        )
    return logits
