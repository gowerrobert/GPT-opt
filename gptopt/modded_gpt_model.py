
import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== Improvements: RMSNorm, SwiGLU, Rotary Embeddings (RoPE), Config flags ======

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).clamp_min(self.eps).sqrt()
        return (x / rms) * self.weight


def build_rope_cache(seq_len: int, dim: int, base: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    half = dim // 2
    freq = torch.arange(half, device=device, dtype=torch.float32) / half
    inv_freq = base ** (-freq)
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)  # (seq_len, half)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D)
    B, H, T, D = x.shape
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:half*2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1,1,T,half)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated


class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, 2 * hidden_dim)
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.proj(x).chunk(2, dim=-1)
        return self.out(F.silu(a) * b)


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    rope: bool = True             # Rotary positional embeddings
    rope_base: int = 10000
    swiglu: bool = True           # SwiGLU feed-forward
    ff_mult: float = 4.0 / 3.0    # Effective expansion (PaLM style ~2.66x vs 4x)
    no_layernorm: bool = False
    flash_attention: bool = True  # Use SDPA (FlashAttention path)
    tie_embeddings: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        if not config.flash_attention:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )
        self.rope_cache = None  # (cos, sin)

    def maybe_init_rope(self, device):
        if self.config.rope and self.rope_cache is None:
            cos, sin = build_rope_cache(self.config.block_size, self.head_dim * 2, self.config.rope_base, device)
            self.rope_cache = (cos, sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.config.rope:
            self.maybe_init_rope(x.device)
            cos, sin = self.rope_cache
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if self.config.flash_attention:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            att = torch.softmax(att, dim=-1)
            att = F.dropout(att, p=self.config.dropout, training=self.training)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        inner = int(config.n_embd * config.ff_mult)
        if config.swiglu:
            self.ff = SwiGLU(config.n_embd, inner)
        else:
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.swiglu:
            x = self.ff(x)
        else:
            x = self.c_fc(x)
            x = F.gelu(x)
            x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        norm = RMSNorm if not config.no_layernorm else nn.Identity
        self.ln1 = norm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = norm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, device):
        super().__init__()
        self.config = config
        self.device = device
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # Use position embedding only if not using RoPE
        self.wpe = nn.Embedding(config.block_size, config.n_embd) if not config.rope else nn.Identity()
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        norm = RMSNorm if not config.no_layernorm else nn.Identity
        self.ln_f = norm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.wte.weight  # weight tying

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Sequence length exceeds block_size"
        tok = self.wte(idx)
        if isinstance(self.wpe, nn.Embedding):
            pos = torch.arange(0, T, device=idx.device)
            pos_emb = self.wpe(pos).unsqueeze(0)
            x = tok + pos_emb
        else:
            x = tok  # RoPE path
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss




# ====== Factory for improved GPT ======
def build_gpt(config_dict, device):
    cfg = GPTConfig(
        vocab_size=config_dict.get("vocab_size", 50257),
        block_size=config_dict.get("block_size", 1024),
        n_layer=config_dict.get("n_layer", 12),
        n_head=config_dict.get("n_head", 12),
        n_embd=config_dict.get("n_embd", 768),
        dropout=config_dict.get("dropout", 0.1),
        rope=config_dict.get("rope", True),
        rope_base=config_dict.get("rope_base", 10000),
        swiglu=config_dict.get("swiglu", True),
        ff_mult=config_dict.get("ff_mult", 4.0 / 3.0),
        no_layernorm=config_dict.get("no_layernorm", False),
        flash_attention=config_dict.get("flash_attention", True),
        tie_embeddings=config_dict.get("tie_embeddings", True),
    )
    model = GPT(cfg).to(device)
    return model
# ...existing code...