# activation_comparison

Does a newer activation function actually help, or is that just what the
paper says? One architecture, one dataset, one activation swapped at a
time — results land side by side so they're comparable.

Self-contained: everything needed to reproduce this lives under this
directory (own `.gitignore`, no imports outside `activation_comparison/`).

## Setup

GPT-Neo with alternating global/local attention -- same design as
TinyStories-1M, but a smaller configuration:

| | TinyStories-1M | here |
|---|---|---|
| vocab | 50257 | 512 (BPE, trained on TinyStories) |
| hidden_size | 64 | 16 |
| num_layers | 8 | 4 |
| num_heads | 16 | 4 (matches TinyStories-1M's head_dim=4, not its raw head count) |
| context_length | 2048 | 128 |

## Results

Eval loss (nats), 3 seeds, mean ± sample-stdev:

| activation | seed 0 | seed 1 | seed 2 | mean ± std | params |
|---|---|---|---|---|---|
| gelu_new | 3.1630 | 3.1622 | 3.1684 | 3.1645 ± 0.0034 | 23200 |
| xielu | 3.1291 | 3.1223 | 3.1396 | 3.1303 ± 0.0087 | 23208 |
| polyrelu | 3.1371 | 3.1299 | 3.1329 | 3.1333 ± 0.0036 | 23216 |
| swiglu | 3.1567 | 3.1669 | 3.1720 | 3.1652 ± 0.0078 | 23152 |
| powlu | 3.1749 | 3.1844 | 3.1702 | 3.1765 ± 0.0072 | 23152 |

- **gelu_new** — GPT-Neo's default.
- **xielu** — Huang & Schlag, [arXiv:2411.13010](https://arxiv.org/abs/2411.13010).
- **polyrelu** — Zhuo et al., ICLR 2025, [arXiv:2411.03884](https://arxiv.org/abs/2411.03884).
- **swiglu** — Shazeer, [arXiv:2002.05202](https://arxiv.org/abs/2002.05202).
- **powlu** — Jiang et al., [arXiv:2605.25704](https://arxiv.org/abs/2605.25704).
