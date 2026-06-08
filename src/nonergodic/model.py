"""Tiny transformer (1 layer, 1 head, 1 MLP) for the non-ergodic Mess3 task."""
import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    """Pre-LN transformer with a single attention head and MLP.

    forward(idx, return_all=True) exposes every intermediate activation:
    embedding, post-attention residual, post-MLP residual, final (LN'd)
    residual that feeds the unembedding, logits, and attention weights.
    """
    def __init__(self, cfg):
        super().__init__()
        d, dm = cfg.d_model, cfg.d_mlp
        self.te = nn.Embedding(cfg.vocab, d)
        self.pe = nn.Embedding(cfg.seq_len, d)
        self.l1 = nn.LayerNorm(d)
        self.at = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True)
        self.l2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, dm), nn.ReLU(), nn.Linear(dm, d))
        self.lf = nn.LayerNorm(d)
        self.un = nn.Linear(d, cfg.vocab, bias=False)
        self.cfg = cfg

    def forward(self, idx, return_all=False):
        Tn = idx.shape[1]
        pos = torch.arange(Tn, device=idx.device)
        emb = self.te(idx) + self.pe(pos)[None]
        h = self.l1(emb)
        mask = torch.triu(torch.ones(Tn, Tn, device=idx.device), diagonal=1).bool()
        attn_out, attn_w = self.at(h, h, h, attn_mask=mask,
                                   need_weights=return_all,
                                   average_attn_weights=False)
        resid_post_attn = emb + attn_out
        resid_post_mlp = resid_post_attn + self.mlp(self.l2(resid_post_attn))
        resid_final = self.lf(resid_post_mlp)          # input to unembed
        logits = self.un(resid_final)
        if return_all:
            return {
                "embedding": emb,
                "resid_post_attn": resid_post_attn,
                "resid_post_mlp": resid_post_mlp,
                "resid_final": resid_final,
                "logits": logits,
                "attn_weights": attn_w,
            }
        return logits
