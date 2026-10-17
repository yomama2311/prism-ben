"""Tiny stand-in for the LeWM JEPA object (same attribute API) used only by the tests."""
import torch
from torch import nn
from types import SimpleNamespace


class StubEncoder(nn.Module):
    def __init__(self, d_hidden=32):
        super().__init__()
        self.proj = nn.Linear(48, d_hidden)

    def forward(self, x, interpolate_pos_encoding=True):
        pooled = nn.functional.adaptive_avg_pool2d(x, 4).flatten(1)      # (B, 48)
        tok = self.proj(pooled).unsqueeze(1)                              # (B, 1, Dh)
        return SimpleNamespace(last_hidden_state=tok)


class StubJEPA(nn.Module):
    def __init__(self, d=16, act_dim=35, d_hidden=32):
        super().__init__()
        self.encoder = StubEncoder(d_hidden)
        self.projector = nn.Linear(d_hidden, d)
        self.action_encoder = nn.Sequential(nn.Linear(act_dim, d))
        self.core = nn.Sequential(nn.Linear(2 * d, 64), nn.GELU(), nn.Linear(64, d))

    def predict(self, emb, act_emb):                                      # (B,T,D),(B,T,D)->(B,T,D)
        return emb + self.core(torch.cat([emb, act_emb], dim=-1))
