# Basically taken from Karpathy's llm.c repo
# https://github.com/karpathy/llm.c/blob/master/train_gpt2.py
import torch
from torch import  nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

# --- RoPE utilities ---
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape(x.shape)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self.cached_seq_len = 0

    def get_embed(self, seq_len: int, device, dtype):
        if self.cos_cached is None or self.cached_seq_len < seq_len:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
            emb = torch.cat((freqs, freqs), dim=-1)            # (T, dim)
            cos = emb.cos()[None, None, :, :]                  # (1,1,T,dim)
            sin = emb.sin()[None, None, :, :]                  # (1,1,T,dim)
            self.cos_cached = cos
            self.sin_cached = sin
            self.cached_seq_len = seq_len
        return (self.cos_cached[:, :, :seq_len, :].to(dtype=dtype, device=device),
                self.sin_cached[:, :, :seq_len, :].to(dtype=dtype, device=device))

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D), cos/sin: (1,1,T,D)
    return (x * cos) + (rotate_half(x) * sin)

# Added: RMSNorm (LLaMA/Mistral)
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash_attention = config.flash_attention
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        # RoPE
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim % 2 == 0, "RoPE requires even head_dim"
        self.rope = RotaryEmbedding(self.head_dim, base=10000.0)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # apply RoPE to q,k
        cos, sin = self.rope.get_embed(T, x.device, x.dtype)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        if self.flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # SwiGLU with 2/3 width of the 4x MLP (PaLM): inner = 8/3 * d_model
        inner = int((4 * config.n_embd) * 2 / 3)
        self.c_fc    = nn.Linear(config.n_embd, inner, bias=False)   # value path
        self.c_gate  = nn.Linear(config.n_embd, inner, bias=False)   # gate path
        self.act     = nn.SiLU()
        self.c_proj  = nn.Linear(inner, config.n_embd, bias=False)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        v = self.c_fc(x)
        g = self.act(self.c_gate(x))
        x = v * g
        x = self.c_proj(x)
        return x

class AttentionBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Switched LayerNorm -> RMSNorm (fall back to Identity if disabled)
        self.ln_1 = RMSNorm(config.n_embd) if not config.no_layernorm else nn.Identity()
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd) if not config.no_layernorm else nn.Identity()
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    no_layernorm: bool = False
    flash_attention: bool = True

@dataclass
class CausalLMOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor

class GPT(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
       
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([AttentionBlock(config) for _ in range(config.n_layer)]),
            # Switched final LayerNorm -> RMSNorm
            ln_f = RMSNorm(config.n_embd) if not config.no_layernorm else nn.Identity(),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator(device="cpu") #self.device)
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else 0.02/math.sqrt(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                labels: torch.Tensor | None = None,
                **kwargs):
        # HF-style API: input_ids instead of idx
        idx = input_ids
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        tok_emb = self.transformer.wte(idx)
        x = tok_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            if labels.dtype != torch.long:
                labels = labels.long()
            # Map user ignore_index -1 -> HF standard -100
            if (labels == -1).any():
                labels = labels.masked_fill(labels == -1, -100)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        return CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self,
                 input_ids: torch.Tensor,
                 max_new_tokens: int,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 top_k: int | None = None,
                 **kwargs):
        # HF-like generate (simplified, no beam/past key values)
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.config.block_size else input_ids[:, -self.config.block_size:]
            out = self(idx_cond)
            logits = out.logits[:, -1, :] / temperature
            if do_sample:
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids
