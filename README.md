
# gpt2-diy

Welcome to **gpt2-diy** — a personal, from-scratch reproduction of GPT-2, inspired by Andrej Karpathy’s video [Let's Reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=11).

This project aims to deeply understand transformer-based language models by rebuilding GPT-2 independently, referencing original papers and consulting existing codebases only lightly when necessary.

---

## Motivation

The best way to learn is by doing.

**gpt2-diy** is a hands-on journey to build skill and acquire deep knowledge of modern language models. By independently reproducing GPT-2, this project hopefully helps with practical understanding of architectures, training dynamics, optimization strategies, and scaling laws.

---

## Inspirations and References

- [Let's Reproduce GPT-2 by Andrej Karpathy](https://www.youtube.com/watch?v=l8pRSuU81PU&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=11)
- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Improving Language Understanding by Generative Pre-Training (Radford et al., 2018) — GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Language Models are Unsupervised Multitask Learners (Radford et al., 2019) — GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Language Models are Few-Shot Learners (Brown et al., 2020) — GPT-3](https://arxiv.org/abs/2005.14165)

---

## Project Plan

A rough outline based on the "Let's Reproduce GPT-2" methodology:

1. **Skim Papers**  
   Collect factual information about architecture design and training strategies.

2. **Load OpenAI Model Weights**  
   Use Hugging Face Transformers to load original GPT-2 weights, providing a reference for our own implementation.

3. **Implement Model in PyTorch**  
   Write the model relying only on the papers. Use the Hugging Face repo solely to copy correct weight names for compatibility.

4. **Model Loading Method (`from_pretrained`)**  
   Add a method to load Hugging Face's GPT-2 weights into the custom model.

5. **Implement Generate Functionality**  
   Add text generation to validate the loaded model weights.

6. **Tiny Shakespeare Data Preparation**

7. **Compute Loss (at Initialization as well)**

8. **Optimization: Check on One Batch**

9. **Data Loader Lite**

10. **Paper Adjustments**  
    - Weight tying between embedding and output projection layers.  
    - Correct weight initialization.

11. **Speed Up Training**  
    - Enable TensorFloat32 (TF32) precision.  
    - Use bf16 where available (Ampere+ GPUs).  
    - Apply `torch.compile` for compilation speedup.  
    - Integrate Flash Attention.  
    - Tune batch sizes for hardware efficiency.

12. **Optimization Settings from GPT-3**  
    - Gradient accumulation to emulate a large effective batch size (~0.5M tokens).

13. **Distributed Data Parallel (DDP)**

14. **Switch to a "Real" Dataset**  
    Move training from Tiny Shakespeare to a FineWeb sample.

15. **Evaluation and Logging**  
    Set up evaluation tasks like HellaSwag for zero-shot benchmarks.

---

## Paper Notes (for all of them)
- GPT-2 uses new dataset of millions of web pages (WebText)
- The capacity of the language model is essential to the success of zero-shot task transfer
- This leads to the path of building language processing systems which learn to perform tasks from their naturally occurring demonstrations.
- Data - web scrape emphasizing document quality, no assumptions about specific downstream tasks
- Follows transformer architecture and gpt-1

**From gpt-1 paper:** 
- 12 layer decoder only with masked self-attention heads 
- 768 dimensional states and 12 attention heads = head size is 64, we compute compatibility function (q @ k) and weighted sum of values (att @ v) for all heads independently and in parallel
- 3072 (768 * 4) dimensional inner state for position-wise feed-forward networks
- Weight init N(0, 0.02). 
- Residual, embedding and attention dropout p=0.1
- Adamw baked L2 regularization w = 0.01 (for non-bias or gain weights)
- GELU non-linearity
- Learned position embeddings
- Adam optimization with max lr = 2.5e-4
- learning rate was increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule
- 100 epochs
- batch size 64
- block size 512

**Gpt-2 modifications:**
- Layer-norm moved to the input of each sub-block (block = attention + feed forward) and one layer-norm after the last self-attention block 
- Modified initialization: scale the weights of residual layers at initialization by a factor of 1/sqrt(N), where N is the number of residual layers
- Vocabulary -> to 50,257
- block size 1024

**Gpt-3:**
- Adam with β1 = 0.9, β2 = 0.95, and  = 10e−8
- Clip the global norm of the gradient at 1.0
- Cosine decay for learning rate down to 10% of its value, over 260 billion tokens (after 260 billion tokens, training continues at 10% of the original learning rate)
- linear LR warmup over the first 375 million tokens (roughly 10%)
- All models were trained for a total of 300 billion tokens

---

> 🚧 Work in Progress: This project is actively being built and improved step-by-step.
