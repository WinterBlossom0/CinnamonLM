# recursion_comparison

Does a small transformer whose middle layers are looped match a transformer with unique layers at the same
effective depth, and does a published loop method do better than the plain loop?

## Setup

GPT-Neo with alternating global and local attention (window 16), hidden size 16, 4 heads, gelu_new, tied
embeddings, context 128, a 512-token BPE tokenizer trained on TinyStories, and TinyStories at a pinned revision
(822,344,370 training and 8,280,406 validation tokens). Every arm uses one protocol: AdamW at learning rate 1e-2
(weight decay 0.01, with the parameters that each loop method's source exempts from decay set to 0), batch 640,
3 epochs (30,114 steps), cosine schedule with 500 warmup steps, gradient clipping at 1.0, fp32, torch.compile,
seeds 0-2. `real_D` has D unique layers. `recursive_xN` has 4 unique layers, and its middle pair runs N times, for
an effective depth of 2 + 2N. The method arms loop the same middle pair 8 times (effective depth 18): re-injection
(Marchenko et al. 2026, arXiv:2610.00673), Hyperloop (Zeitoun et al. 2026, arXiv:2604.21254), Parcae (Prairie et
al. 2026, arXiv:2604.12946), the Gated Recurrent Transformer (Hegazy et al. 2026, arXiv:2608.15062) and MeSH (Yu et
al. 2026, arXiv:2510.07739).

## Results

Eval loss (nats) on the full validation split after epoch 3, ranked by the mean of 3 seeds (mean ± sample std):

| arm | unique layers | effective depth | params | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|---|---|---|
| real_18 | 18 | 18 | 68448 | 2.3105 | 2.3336 | 2.3166 | 2.3202 ± 0.0120 |
| real_10 | 10 | 10 | 42592 | 2.4406 | 2.5210 | 2.4212 | 2.4609 ± 0.0529 |
| recursive_x8 | 4 | 18 | 23200 | 2.9401 | 2.7812 | 2.8140 | 2.8451 ± 0.0839 |
| real_6 | 6 | 6 | 29664 | 2.8571 | 2.8598 | 2.8609 | 2.8593 ± 0.0020 |
| hyperloop_x8 | 4 | 18 | 30104 | 2.9040 | 2.8959 | 2.9126 | 2.9042 ± 0.0084 |
| recursive_x2 | 4 | 6 | 23200 | 2.9462 | 2.9432 | 2.9447 | 2.9447 ± 0.0015 |
| recursive_x4 | 4 | 10 | 23200 | 2.9627 | 2.9475 | 2.9578 | 2.9560 ± 0.0077 |
| grt_x8 | 4 | 18 | | | | | pending rerun (results from earlier code) |
| parcae_x8 | 4 | 18 | | | | | pending rerun (results from earlier code) |
| reinject_x8 | 4 | 18 | | | | | pending rerun (results from earlier code) |
| mesh_x8 | 4 | 18 | | | | | pending rerun (results from earlier code) |

## Limits

- Architectures only: Parcae's per-sequence depth sampling and truncated backpropagation, the Gated Recurrent
  Transformer's random loop count and re-injection's exit gate are left out.
- One fixed protocol for all arms; no arm has its own hyperparameters.
- No bit reproducibility: the compiled backward adds with atomics in a random order, so a rerun of a seed does not
  repeat its numbers exactly.
