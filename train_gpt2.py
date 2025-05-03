"""
GPT-2 Implementation, training and evaluation.
Inspired by Andrej Karpathy's great "Let's reproduce GPT-2"
https://youtu.be/l8pRSuU81PU?si=9hpD5HfBAfzbNUNb
"""

from dataclasses import dataclass
import math
import time

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

# Module Definitions
##############################################
@dataclass
class GPT2Config:
  n_layers = 12
  n_embd = 768
  n_heads = 12
  block_size = 1024
  vocab_size = 50264 # divisible by 8

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
    # use native flash attention, see: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
    # ensure shared weights as in paper
    self.transformer['wte'].weight = self.lm_head.weight
    # apply gpt-2 weight initialization
    self.apply(self._init)
    # scale residual layer weights by sqrt(n_layers * 2)
    # [every sub-block feeding into residual stream]
    scale = 1 / math.sqrt(config.n_layers * 2)
    with torch.no_grad():
      for block in self.transformer['h']:
        block.attn.c_proj.weight.mul_(scale)
        block.mlp.c_proj.weight.mul_(scale)

  def _init(self, m):
    """GPT-2 like initialization"""
    # Note: nn.init does initialization with no_grad context by default
    if isinstance(m, nn.Linear): # all Linear and Embedding weights
      nn.init.normal_(m.weight, 0, 0.02)
      if m.bias is not None:
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.Embedding):
      nn.init.normal_(m.weight, 0, 0.02)

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

##############################################

# Helpers:

# Sample from model to generate some new tokens
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

# Simple dataloader
class DataLoaderLite():
  """Loads and tokenizes tiny dataset, produces batches of data"""
  def __init__(self):
    # initialize gpt-2 tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')
    # create a batch of data
    with open('input.txt', 'r', encoding='utf-8') as file:
      text = file.read()
    tokens = tokenizer.encode(text)
    self.tokens = torch.LongTensor(tokens)
    self.start = 0

  def next_batch(self, batch_size, block_size):
    """Create a batch of training examples and respective targets,
    discard last batch, if it has < batch_size * block_size tokens"""
    n = batch_size * block_size
    data = self.tokens[self.start: self.start + n].view(batch_size, block_size)
    target = self.tokens[self.start+1: self.start + n + 1].view(batch_size, block_size)
    # move offset
    self.start += n
    # reset to 0, if we can't build full batch next iteration
    if (self.start + n + 1) > len(self.tokens):
      self.start = 0
    return data, target

# Setup Optimizer
def setup_optimizer(params):
  # we don't want to apply weight decay to gains and biases
  # we want opportunity to learn scale and shift and not force them towards zero
  wd_params = [p for p in params if p.ndim == 2]
  nowd_params = [p for p in params if p.ndim == 1]
  optimizer = torch.optim.AdamW([{'params': wd_params, "weight_decay": 0.1}, 
                                 {'params': nowd_params, "weight_decay": 0.0}],
                                lr=3e-4, betas=(0.9, 0.95), eps=10e-8,
                                fused=device=='cuda')
  return optimizer

# Simple lr cosine annealing scheme
def get_lr(step, total_steps, max_lr):
  """Simple impl of cosine lr decay with
  linear warmup during first 10% of steps"""
  warmup_steps = total_steps // 10
  decay_steps = total_steps - warmup_steps
  if step < warmup_steps:
    return max_lr * (step + 1) / warmup_steps
  else:
    t = step - warmup_steps
    return 0.5 * max_lr * (1 + math.cos(math.pi * (t / decay_steps)))

##################################################
          
# Main routine with a guard, as some parts can be importable w/o running script
##################################################
if __name__ == "__main__":
  # set random seed
  torch.manual_seed(42)
  # random init of model weights
  config = GPT2Config()
  model = GPT2(config)

  # training related hyperparameters
  eff_batch_size = 512 # it is nice number and 512 * 1024 ~= 0.5mln tokens
  batch_size = 4       # can fit into gpu, adjust accordingly
  # number of steps to accumulate gradients before optimization step
  num_accum_steps = eff_batch_size // batch_size
  num_steps = 2048     # adjust according to the task
  max_lr = 6e-4        # gpt-3 paper for small model
  num_tokens_per_step = num_accum_steps * batch_size * config.block_size

  print(f'Training with effective batch size: {eff_batch_size}, number of accumulation steps: {num_accum_steps}')

  # setting device
  device = 'cpu' 
  if torch.cuda.is_available():
    device = 'cuda'
  if torch.backends.mps.is_available():
    device = 'mps' 
  print(f'using {device=}')

  torch.set_float32_matmul_precision('high') # less bits for internal to matmul float repr

  # use torch compile if on cuda (mps doesn't work for now)
  if device == 'cuda':
    model.compile()
  # set to training mode and put on device
  model.train()
  model.to(device)
  
  # initialize dataloader
  dataloader = DataLoaderLite()
  # optimizer 
  optimizer = setup_optimizer(model.parameters())
  optimizer.zero_grad()
  # accumulators for logging
  start = time.time() # time spent per effective batch
  l = 0.0  # loss per effective batch
  # optimization loop
  for i in range(num_steps):
    x, y = dataloader.next_batch(batch_size, config.block_size)
    x, y = x.to(device), y.to(device)

    # enables autocast
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
      # do forward pass and compute loss
      logits, loss = model(x, y)
    # compute gradients
    loss.backward()
    # clip the global norm of the gradient
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # wait for cuda work to finish
    if device == 'cuda':
      torch.cuda.synchronize()

    l += loss.item() / num_accum_steps # scale by number of accumulation steps as we want average loss for effective batch
    # optimization step ones per number of accumulation steps
    if (i + 1) % num_accum_steps == 0:
      # assign lr according to schedule - we only use lr during optimization step (when we update weights)
      lr = get_lr(i, num_steps, max_lr)
      for g in optimizer.param_groups:
        g['lr'] = lr
      # do optimization step
      optimizer.step()
      optimizer.zero_grad()
      t = time.time() - start
      # tokens/s rounded
      tps = num_tokens_per_step / t
      print(f'iter: {i} | loss: {l:.4f} | lr: {lr:.2E} | time: {t:.4f}s | {tps:.0f} tok/s |')
      # reset accumulators
      start = time.time()
      l = 0.0
  