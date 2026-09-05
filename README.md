# attention_comparison

Does swapping softmax attention for a linear-attention mechanism change
anything at this scale? Same data, tokenizer, and activation (gelu_new) as
`activation_comparison/`; only the attention mechanism differs.

Self-contained: everything needed to reproduce this lives under this
directory (own `.gitignore`, no imports outside `attention_comparison/`).

## Setup

| | activation_comparison (softmax) | KDA | OSDN | Parallax |
|---|---|---|---|---|
| attention | GPT-Neo global/local softmax | Kimi Delta Attention (gated delta rule) | delta rule + online key preconditioner | softmax + local-covariance correction |
| position embedding | learned absolute | none | none | learned absolute |
| activation | gelu_new | gelu_new | gelu_new | gelu_new |
| d_model | 16 | 16 | 16 | 16 |
| num_layers | 4 | 4 | 4 | 4 |
| context_length | 128 | 128 | 128 | 128 |
| vocab | 512 | 512 | 512 | 512 |
| params | 23200 | 23168 | 23252 | 23212 |

- **KDA** — Kimi Team, "Kimi Linear: An Expressive, Efficient Attention
  Architecture," [arXiv:2510.26692](https://arxiv.org/abs/2510.26692). This
  is the KDA layer type used in every layer, not the paper's actual proposed
  architecture (a 3:1 KDA:MLA hybrid) -- see kda/attention.py's docstring
  for the full list of simplifications from the paper's q/k/v/gate pipeline.
- **OSDN** — Zhou, Li, Liu, Lin, Ge, Ye, "Improving Delta Rule with Provable
  Online Preconditioning in Linear Attention,"
  [arXiv:2605.13473](https://arxiv.org/abs/2605.13473). This is vanilla OSDN
  (Algorithm 1), not OSDN-APF, the more stable variant the paper's own
  larger-scale experiments actually use. The paper's own stated limitations
  say its gains are recall-specific, not general perplexity improvement --
  see osdn/attention.py's docstring.
- **Parallax** — Zuo, Pai, Zeng, Dewulf, Hu, Wang, "Parallax: Parameterized
  Local Linear Attention for Language Modeling,"
  [arXiv:2605.29157](https://arxiv.org/abs/2605.29157). Unlike KDA/OSDN this
  is O(T^2) (full softmax attention plus a covariance correction), not a
  cheaper linear-attention mechanism -- included as a quality-only
  comparison, not an efficiency one. The paper reports Parallax's gains are
  optimizer-dependent: consistent under Muon, but "shrink markedly or even
  disappear" under AdamW, with their own ablation showing the model learns
  to suppress the correction term under AdamW. This project trains with
  AdamW -- see parallax/attention.py's docstring before interpreting any
  Parallax result here as evidence about the mechanism itself.

KDA's and OSDN's chunked forward, and Parallax's efficient streaming form,
are each verified against their own reference computation at the head/dim
configuration actually trained, to float32 precision. Each Triton kernel
(KDA: gate preprocessing, cross-chunk scan, RMSNorm; OSDN: preconditioner
sweep) is separately verified forward and backward against its plain-PyTorch
equivalent at the same configuration.

## Results

Results pending. Two prior confounds were found and fixed: a weight-init
mismatch against the softmax baseline (this project used PyTorch's default
init; activation_comparison uses GPT-Neo's N(0, 0.02)), and a
softmax-baseline config bug (head_dim=1, and a local attention window that
never activated at this context length). The shared MLP activation was also
switched from PolyReLU to gelu_new, matching activation_comparison's own
default/baseline exactly rather than one of its ablation arms. KDA is being
retrained under this corrected configuration; OSDN and Parallax still need
retraining under it too before their numbers are comparable.
