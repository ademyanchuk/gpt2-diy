"""GPT-2 Implementation"""

from dataclasses import dataclass
import math

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

device = 'cpu' 
if torch.cuda.is_available():
  device = 'cuda'
if torch.backends.mps.is_available():
  device = 'mps' 
print(f'using {device=}')

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
    att.masked_fill_(self.bias[:T, :T] == 0.0, -torch.inf)
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

  def forward(self, x, target=None):
    # x shape (B, T)
      B, T = x.shape
      te = self.transformer['wte'](x) # -> (B,T,H)
      pe = self.transformer['wpe'](torch.arange(T, device=x.device)) # -> (T,H)
      x = te + pe # broadcast -> (B,T,H)
      for block in self.transformer['h']:  # -> (B,T,H)
         x = block(x)
      x = self.transformer['ln_f'](x)  # -> (B,T,H)
      logits = self.lm_head(x)  # -> (B,T,V)
      loss = None
      # compute loss if we have targets
      if target is not None:
        loss = F.cross_entropy(logits.view(B*T, -1), target.view(B*T,))
      return logits, loss

  @classmethod
  def from_pretrained(cls, config: GPT2Config):
    # we only use 124m gpt-2 model for now
    # if config differs from gpt2 (124m) model config this will crash
    # init our model
    model = cls(config)
    hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
    # assert all parameter names are equal
    mod_sd, hfmod_sd = model.state_dict(), hf_model.state_dict()
    assert mod_sd.keys() == hfmod_sd.keys(), print('Our and HF model parameter names must be equal!')
    with torch.no_grad():
      # iterate keys (param names) and copy hf to ours
      for k in mod_sd.keys():
        # check if Conv1d weight, this one need transpose
        if "c_" in k and k.endswith("weight"):
          mod_sd[k].copy_(hfmod_sd[k].T)
        else:
          mod_sd[k].copy_(hfmod_sd[k])

    return model

def generate(inp, model, block_size, max_new_tokens):
  # Generate `max_new_tokens` based on the input `inp`,
  # using provided model. Returns concatenation of
  # inp and newly generated tokens
  for _ in range(max_new_tokens):
    with torch.no_grad():
      # clamp input to block size from left
      logits = model(inp[:,-block_size:])
      # only need probs for the last time step
      logits = logits[:, -1, :]
      probs = torch.softmax(logits, -1)
      topk_vals, topk_ids = torch.topk(probs, k=50) # topk_ids are indices of original tensor `probs`
      topk_sample = torch.multinomial(topk_vals, num_samples=1) # topk_sample are indices of tensor `topk_vals`
      new_id = torch.gather(topk_ids, -1, topk_sample) # map sampled ids to original `probs` ids
      inp = torch.cat((inp, new_id), dim=-1)
  return inp
          

if __name__ == "__main__":
  # set random seed
  torch.manual_seed(42)
  # load our model from pretrained weights
  config = GPT2Config()
  model = GPT2.from_pretrained(config)
  model.to(device)
  model.eval()
  # create input and tokenize it
  text = "Hello, I'm a language model,"
  tokenizer = tiktoken.get_encoding('gpt2')
  ids = tokenizer.encode(text)
  # convert to tensor, replicate and move to cuda
  inp = torch.LongTensor(ids).unsqueeze(0).repeat(5, 1)
  inp = inp.to(device)
  # generate, decode, and print
  out = generate(inp, model, config.block_size, 50)
  out = tokenizer.decode_batch(out.data.tolist())
  for res in out:
    print(res)
    print('------------')
