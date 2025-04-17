"""GPT-2 Implementation"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn

@dataclass
class GPT2Config:
  n_layers = 12
  n_embd = 768
  n_heads = 12
  block_size = 1024
  vocab_size = 50257

class Attention(nn.Module):
  """Scaled Self Attention"""
  def __init__(self, n_embd, n_heads) -> None:
    super().__init__()
    assert n_embd % n_heads == 0, print(f"{n_embd=} % {n_heads=} must == 0!")
    self.n_heads = n_heads
    # one set of weights which includes query, key, value weights
    # n_embd = head_size * n_heads
    self.c_attn = nn.Linear(n_embd, n_embd * 3) # explicitly (H, HS*NH*3)
    self.c_proj = nn.Linear(n_embd, n_embd)
  
  def forward(self, x):
    B,T,H = x.shape
    Hs = H // self.n_heads # head size
    # set of queries, keys and values packed together
    qkv = self.c_attn(x) # -> (B,T,H*3)
    # split them
    q,k,v = torch.split(qkv, H, dim=-1) # -> 3 (B,T,H)
    # we want the compatibility function to be computed for 
    # all heads independently, so we need n_heads (Nh)
    # as a separate dimension, and transpose to apply 
    # compatibility and weighting functions on appropriate dimensions
    q = q.view((B,T,self.n_heads,Hs)).transpose(1, 2) # (B,Nh,T,Hs)
    k = k.view((B,T,self.n_heads,Hs)).transpose(1, 2) # (B,Nh,T,Hs)
    v = v.view((B,T,self.n_heads,Hs)).transpose(1, 2) # (B,Nh,T,Hs)
    # compute tokens compatibility (affinity)
    att = q @ k.transpose(-2, -1) # -> (B,Nh,T,T)
    att.div_(math.sqrt(Hs)) # TODO


class Block(nn.Module):
  def __init__(self, n_embd) -> None:
    super().__init__()
    self.ln_1 = nn.LayerNorm(n_embd)
    self.attn = Attention(n_embd)
    self.ln_2 = nn.LayerNorm(n_embd)
    self.mlp = MLP(n_embd)
  
  def forward(self, x):
    # x shape (B,T,H)
    # layer norm applied to the input of each sub-block
    x = x + self.attn(self.ln_1(x)) # -> (B,T,H)
    x = x + self.mlp(self.ln_2(x)) # -> (B,T,H)

class GPT2(nn.Module):
  def __init__(self, config: GPT2Config):
    super().__init__()
    self.transformer = nn.ModuleDict({
      'wte': nn.Embedding(config.vocab_size, config.n_embd), # (V, H)
      'wpe': nn.Embedding(config.block_size, config.n_embd), # (T, H)
      'h': nn.ModuleList([Block(config.n_embd) for _ in range(config.n_layers)]), # (H, H)
      'ln_f': nn.LayerNorm(config.n_embd) # (H, H)
    })
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # (H, V)

  def forward(self, x):
    # x shape (B, T)
      B, T = x.shape
      te = self.transformer['wte'](x) # -> (B,T,H)
      pe = self.transformer['wpe'](torch.arange(T, device=x.device)) # -> (T,H)
      x = te + pe # broadcast -> (B,T,H)
      for block in self.transformer['h']:  # -> (B,T,H)
         x = block(x)
      x = self.transformer['ln_f'](x)  # -> (B,T,H)
      logits = self.lm_head(x)  # -> (B,T,V)
      return logits
