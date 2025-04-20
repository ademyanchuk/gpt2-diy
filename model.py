"""GPT-2 Implementation"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPT2Config:
  n_layers = 12
  n_embd = 768
  n_heads = 12
  block_size = 1024
  vocab_size = 50257

class Attention(nn.Module):
  """Scaled Self Attention"""
  def __init__(self, config: GPT2Config) -> None:
    super().__init__()
    assert config.n_embd % config.n_heads == 0, print(f"{config.n_embd=} % {config.n_heads=} must == 0!")
    self.n_heads = config.n_heads
    # one set of weights which includes query, key, value weights
    # n_embd = head_size * n_heads
    self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3) # explicitly (H, HS*NH*3)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    # low triangle matrix for masking, don't have to be part of model's state dict
    self.register_buffer('bias', torch.tril(torch.ones((config.block_size, config.block_size))), persistent=False)
  
  def forward(self, x):
    B,T,H = x.shape
    Hs = H // self.n_heads # head size [dk in the Attention is all you need paper]
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
    # (B,Nh,T,Hs) @ (B,Nh,Hs,T)
    att = q @ k.transpose(-2, -1) # -> (B,Nh,T,T)
    att.div_(math.sqrt(Hs)) # scale by sqrt(Hs)
    # prevent communication with future tokens, -inf -> 0 after softmax
    att.masked_fill_(self.bias == 0.0, -torch.inf)
    # convert affinities into weights
    att = F.softmax(att, dim=-1) # -> (B,Nh,T,T)
    # compute weighted sum of values
    # (B,Nh,T,T) @ (B,Nh,T,Hs)
    out = att @ v # -> (B,Nh,T,Hs)
    # transpose back and concat
    # (B,Nh,T,Hs) -> (B,T,Nh,Hs) -> (B,T,H)
    out = out.transpose(1, 2).reshape((B,T,H))
    # project through linear
    return self.c_proj(out) # -> (B,T,H)

class MLP(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.c_fc = nn.Linear(n_embd, n_embd*4)
    self.c_proj = nn.Linear(n_embd*4, n_embd)
    self.act = nn.GELU(approximate="tanh") # as in open ai original

  def forward(self, x):
    # x shape (B,T,H)
    x = self.act(self.c_fc(x)) # -> (B,T,H*4)
    return self.c_proj(x) # -> (B,T,H)


class Block(nn.Module):
  def __init__(self, config: GPT2Config) -> None:
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = Attention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config.n_embd)
  
  def forward(self, x):
    # x shape (B,T,H)
    # layer norm applied to the input of each sub-block
    x = x + self.attn(self.ln_1(x)) # -> (B,T,H)
    x = x + self.mlp(self.ln_2(x)) # -> (B,T,H)
    return x

class GPT2(nn.Module):
  def __init__(self, config: GPT2Config):
    super().__init__()
    self.transformer = nn.ModuleDict({
      'wte': nn.Embedding(config.vocab_size, config.n_embd), # (V, H)
      'wpe': nn.Embedding(config.block_size, config.n_embd), # (T, H)
      'h': nn.ModuleList([Block(config) for _ in range(config.n_layers)]), # (H, H)
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

if __name__ == "__main__":
  config = GPT2Config()
  model = GPT2(config)
  model.to('cuda')
  model.eval()
  print(model)
  x = torch.randint(config.vocab_size, size=(4, config.block_size)).type(torch.LongTensor)
  x = x.to('cuda')
  print(x.shape, x.dtype, x.device)
  out = model(x)
  print(out.shape, out.dtype, out.device)