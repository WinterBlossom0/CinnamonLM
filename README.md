# activation_comparision

Does a newer activation function actually help, or is that just what the
paper says? One architecture, one dataset, one activation swapped at a
time — results land side by side so they're comparable.

Self-contained: everything needed to reproduce this lives under this
directory (own `.gitignore`, no imports outside `activation_comparision/`).

## Setup

GPT-Neo (TinyStories-1M architecture, shrunk for fast iteration):

| | TinyStories-1M | here |
|---|---|---|
| vocab | 50257 | 512 (BPE, trained on TinyStories) |
| hidden_size | 64 | 16 |
| num_layers | 8 | 4 |
| num_heads | 16 | 16 |
| context_length | 512 | 128 |

## Results

Eval loss (nats), 3 seeds, mean ± sample-stdev:

| activation | seed 0 | seed 1 | seed 2 | mean ± std | params |
|---|---|---|---|---|---|
| gelu_new | 3.2324 | 3.2504 | 3.2509 | 3.2446 ± 0.0105 | 23200 |
| xielu | 3.2050 | 3.2141 | 3.2224 | 3.2138 ± 0.0087 | 23208 |
| polyrelu | 3.1946 | 3.2053 | 3.2167 | 3.2055 ± 0.0110 | 23216 |
| swiglu | 3.2303 | 3.2365 | 3.2296 | 3.2321 ± 0.0038 | 23152 |
| powlu | 3.2428 | 3.2512 | 3.2426 | 3.2456 ± 0.0049 | 23152 |

- **gelu_new** — GPT-Neo's default.
- **xielu** — Huang & Schlag, [arXiv:2411.13010](https://arxiv.org/abs/2411.13010).
- **polyrelu** — Zhuo et al., ICLR 2025, [arXiv:2411.03884](https://arxiv.org/abs/2411.03884).
- **swiglu** — Shazeer, [arXiv:2002.05202](https://arxiv.org/abs/2002.05202).
- **powlu** — Jiang et al., [arXiv:2605.25704](https://arxiv.org/abs/2605.25704).
