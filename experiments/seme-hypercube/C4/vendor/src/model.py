"""PathStarTransformer model definition."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, max_seq_len: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).split(C, dim=-1)
        q, k, v = [t.view(B, T, self.n_head, self.d_head).transpose(1, 2) for t in qkv]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class FFN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=True)
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, max_seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class PathStarTransformer(nn.Module):
    """
    Minimal decoder-only transformer for the path-star task.

    Geometry contract:
      - self.emb.weight is L2-normalized per-row in forward (unit sphere geometry).
      - self.pos_emb is NOT normalized. All geometric metrics operate on self.emb.weight only.
      - self.lm_head is untied from self.emb (independent decoding).
      - Monitor pe_norm_ratio() each eval step; flag results if > 3.0.
    """

    def __init__(self, vocab_size: int, d_model: int, n_head: int, n_layer: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, max_seq_len) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        # lm_head is weight-tied to emb: shares emb.weight, no separate parameter.
        # Input embedding uses F.normalize(emb.weight) (unit sphere geometry).
        # Output projection uses raw emb.weight — different effective operation.
        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
        n_layer = len(self.blocks)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, std=std / math.sqrt(n_layer))
            nn.init.normal_(block.ffn.fc2.weight, std=std / math.sqrt(n_layer))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        # Weight-tied output projection (matches Stage 1 checkpoint).
        return F.linear(x, self.emb.weight)

    def pe_norm_ratio(self) -> float:
        """Mean L2 norm of PE vectors. Values > 3.0 indicate PE is dominating inputs."""
        return self.pos_emb.weight.norm(dim=-1).mean().item()

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
