# Methods that improve recursion-based transformers

Literature notes for the recursion proxy (GPT-Neo on TinyStories, hidden 16, 4 heads of size 4, global/local
attention with window 16, context 128, vocab 512, GELU MLP 16->64->16, pre-LayerNorm, final LayerNorm, learned
absolute position embedding, tied embeddings, AdamW lr 1e-2, batch 640, 3 epochs, gradient clip 1.0, 23,200
parameters; looped arms run layer 1, then layers 2-3 r times, then layer 4, with plain weight sharing).

Each paper below was read from its arXiv PDF text (downloaded from arxiv.org/pdf/<id> and converted with
pdftotext) or from its arXiv HTML page. Papers marked "(read by extraction agent)" were read from the same PDF
text by a helper agent; I spot-checked the key numbers of those papers against the PDF text (checked values are
marked "[checked]"). Tables that were garbled in the PDF text are flagged. Nothing is cited from memory.

Parameter arithmetic for the proxy (my own count, it reproduces the 23,200 total): token embedding 512x16 =
8,192; position embedding 128x16 = 2,048; one GPT-Neo layer = 3,232 (two LayerNorms 64, q/k/v 768 without bias,
out-projection 272, MLP 2,128); final LayerNorm 32. One percent of the budget is about 232 parameters.

Cause codes used below:
1. fewer unique parameters (capacity);
2. each pass applies the identical function (no depth specialization);
3. hidden state drift / norm growth / instability / seed sensitivity;
4. the input is forgotten across passes (no input injection);
5. optimization through many shared passes (gradient explosion or vanishing, truncated backprop);
6. the same compute for every token (no adaptive depth);
7. train/test mismatch of the loop count.

Our proxy uses a fixed loop count in training and evaluation, so cause 7 does not apply to the current arms.

---

## 0. The size of the problem: looped vs unique layers

### Saunshi et al. 2025 — Reasoning with Latent Thoughts: On the Power of Looped Transformers
- Citation: N. Saunshi, N. Dikkala, Z. Li, S. Kumar, S. J. Reddi. ICLR 2025. arXiv:2502.17416.
  https://arxiv.org/abs/2502.17416
- Method: a k-layer block applied L times with shared weights, written (k⊗L); "middle looping" keeps the first
  and last layers unique and loops the middle block, e.g. (4⊗1,4,1). Plain looping: the paper describes no input
  injection and no normalization change. It also proposes a "looping-inspired regularizer" for a non-looped
  model: R_G(k) = (1/(L−k)) Σ_i Σ_j Cosine(θ^(ik+j), θ^((i+1)k+j)), total loss L_xent + λ_reg·mean_G R_G(k).
- Causes: documents cause 1/2 (perplexity gap); the regularizer trades in the opposite direction (makes a
  unique-layer model behave like a looped one).
- Evidence: Pile, 250B tokens, 24-layer baseline (dimension 2048), batch 512, length 1280, peak lr 0.01.
  Perplexity: 24⊗1 7.40; 12⊗1 8.16, 12⊗2 7.90, middle loop (12x params, 24x FLOPs) 7.81; 8⊗1 8.75, 8⊗3 8.19;
  6⊗1 9.25, 6⊗4 8.42; 4⊗1 10.12, 4⊗6 8.79. Looping closes 34, 41, 44, 48% of the perplexity gap for 12, 8, 6, 4
  layers. Middle looping had better perplexity than plain looping (7.81 vs 7.90). Regularizer on the 24-layer
  model: k=4, λ=10 gives perplexity 7.38 vs 7.40 and better reasoning averages (30.0 vs 26.0).
  Synthetic tasks: 3 seeds per setting, maximum accuracy reported.
- Cost: regularizer is free in parameters; looping costs L times the FLOPs.
- Fit: our arms already reproduce the paper's pattern (small gain over the 4-layer base, far from unique layers).
  The regularizer does not fit: it applies to a model with more unique parameters, not to a 23,200-parameter arm.
  Middle looping is what our arms already do.

### Bae et al. 2025 — Mixture-of-Recursions (sharing-strategy part)
- Citation: S. Bae, Y. Kim, R. Bayat, S. Kim, J. Ha, T. Schuster, A. Fisch, H. Harutyunyan, Z. Ji,
  A. Courville, S.-Y. Yun. arXiv:2507.10524. https://arxiv.org/abs/2507.10524
- Evidence on sharing order (Appendix F, Table 10/11, models trained from scratch on 10B FineWeb-Edu tokens):
  "Middle-Cycle is consistently the safest choice"; at 360M it has the lowest NLL for 2 and 3 recursions;
  at 135M, Cycle is slightly ahead at Nr=2 (3.0071 vs 3.0330) but Middle-Cycle wins at Nr=3 (3.1048 vs 3.1154);
  "pure Sequence sharing records the worst NLL in all four settings". At 730M with equal unique parameters,
  Middle-Cycle (1+8×3+1) beats Cycle (10×3) on NLL (2.7552 vs 2.7573).
  Main table (360M class): vanilla 315M non-embedding params NLL 2.7824; recursive Middle-Cycle Nr=2 167M 2.8079;
  Nr=3 118M 2.8466.
- Fit: our layer order [1, (2,3)×r, 4] is Middle-Cycle, the order this paper found best. See section 6 for the
  router.

### Takase & Kiyono 2023 — Lessons on Parameter Sharing across Layers in Transformers
- Citation: S. Takase, S. Kiyono. SustaiNLP 2023. arXiv:2104.06022. https://arxiv.org/abs/2104.06022
- Method: M unique layers stacked to N layers by Sequence (each set repeated on consecutive layers: AABBCC),
  Cycle (ABCABC) or Cycle (rev) (ABC...CBA at the end).
- Cause: 1 (spend the parameter budget on wider layers, then share).
- Evidence: WMT16 En-De, M=6: vanilla 6 layers 61M 26.48 BLEU; Universal (1 layer ×6, 63M) 26.90; N=18:
  Sequence 27.02, Cycle 27.25, Cycle (rev) 27.31. WikiText-103 (121M, pre-LN): test perplexity vanilla 21.13,
  Universal 23.84, Sequence 19.69, Cycle 19.69, Cycle (rev) 20.24. LibriSpeech (pre-LN): Sequence best
  (3.32/7.71 WER). The authors say the best strategy depends on task and architecture and recommend Sequence as a
  first step. No instability reported.
- Fit: zero-parameter reordering. The evidence conflicts with MoR (Sequence worst there). Not a priority arm.

### Nguyen & Lin 2025 — Intra-Layer Recurrence in Transformers for Language Modeling
- Citation: A. Nguyen, W. Lin. arXiv:2505.01855. https://arxiv.org/abs/2505.01855
- Method: a reuse map R = [r1,...,rL] applies layer l r_l times in place (Sequence-style recurrence per layer).
  Block recurrence baseline uses additive injection x = h + e.
- Causes: 1, 2.
- Evidence: LLaMA-style 1.2M-parameter model (hidden 128, 4 layers, vocab 1,024), 500M FineWeb-Edu tokens,
  AdamW lr 3e-3, batch 64, clip 1.0. Perplexity (NoPE/RoPE/Learned/ALiBi): baseline 16.57/15.56/14.98/14.38;
  block recurrence r=2 15.29/14.12/14.27/14.23; [4,2,1,1] 14.64/13.77/14.20/13.63; [1,1,1,2] 16.98/15.02/14.94/14.23.
  "Applying recurrence to earlier layers usually yields the largest perplexity gains." 100M model: one map only
  ([1,2,1,...,1]), gains of about 0.1 perplexity. Single seed (no variance reported).
- Cost: zero parameters; FLOPs grow with the reuse map.
- Fit: the closest scale to ours (1.2M params, 4 layers). Note for the proxy: with learned absolute positions,
  block recurrence and the best ILR maps differ by under 0.1 perplexity, so the placement of the loop is a
  second-order effect at this scale.

---

## 1. Cause 1 — fewer unique parameters (capacity)

Pattern (Saunshi; Ouro Appendix D; our runs): looped models gain over the iso-parameter base but stay well
behind unique-layer models of the same depth, and the gap grows with loop count and shrinks with model size.
More loops of the same block give diminishing returns (Hyperloop Fig. 2; Ouro T=8 run in "Closing the Loop").

Methods that add a small number of loop-specific parameters:

### Bae et al. 2025 — Relaxed Recursive Transformers (per-loop LoRA)
- Citation: S. Bae, A. Fisch, H. Harutyunyan, Z. Ji, S. Kim, T. Schuster. ICLR 2025. arXiv:2410.20672.
  https://arxiv.org/abs/2410.20672
- Method: Cycle-shared layers; each shared layer gets a separate LoRA module per loop. LoRA is initialized from a
  truncated SVD of the difference between the original pretrained layer and the shared layer. Rank 0 is the pure
  recursive model.
- Causes: 1, 2.
- Evidence: uptraining from pretrained checkpoints, not training from scratch. Recursive Gemma 1B (0.99B, 15B
  tokens, Stepwise init) 51.7 few-shot average vs 38.2 for a same-size Gemma trained from scratch. Relaxed Gemma
  rank 512 (1.60B params, 60B tokens, distillation) 58.4 vs full-size uptrained 58.6. Parameter overhead: 0.99B at
  rank 0, 1.07B at rank 64, 1.15B at rank 128, 1.30B at rank 256, 1.60B at rank 512. All pretrained-weight
  initializations beat random initialization.
- Fit: weak. The gains come from pretrained initialization; there is no from-scratch LoRA result. Independent
  from-scratch evidence is negative: Hyperloop (below) Table 7, LoRA rank 4/8/16/32 on a 135M looped model gives
  perplexity 14.85/14.81/14.80/14.77 vs 14.85 at rank 0; "Closing the Loop" finds rank-32 step LoRA "provide[s]
  little benefit" on a converted Qwen3-1.7B. Cost at our scale: rank-1 LoRA on all 6 matrices of both shared
  layers costs 576 parameters per extra pass (4,032 for x8, 17%).

### Zeitoun, Torroba-Hennigen, Kim 2026 — Hyperloop Transformers
- Citation: A. Zeitoun, L. Torroba-Hennigen, Y. Kim. arXiv:2604.21254 (v3, July 2026).
  https://arxiv.org/abs/2604.21254
- Method: begin / middle / end blocks, only the middle block is looped (Middle-Cycle). The residual stream is
  expanded to n parallel streams; after each loop (not after each layer) loop-specific, input-dependent matrices
  H_pre (read n streams into one), H_post (write the block output to the n streams) and a diagonal sigmoid
  H_res (stream retention) mix the streams, and a learned "loop position embedding" e_l is added:
  y(l+1) = H_res·y(l) + H_post·F(H_pre·y(l)) + e_l. The streams are averaged before the end block.
- Causes: 1, 2 (loop-specific parameters), 3 (gated retention).
- Evidence: FineWeb-Edu, 2.5x Chinchilla tokens, AdamW, n=4. Perplexity: 238M Transformer 14.65, looped
  [2L→4L×3→2L] 135.5M 14.85, Hyperloop 135.7M 14.40; 990M Transformer 10.19, looped 579M 10.02, Hyperloop 9.65;
  2.0B Transformer 8.60, looped 990.5M 8.68, Hyperloop 8.49. Hyperloop needs 150-300K extra parameters.
  Ablations at 135M: hyper-connections every loop 14.40 vs every layer 14.45 vs once 14.63; H_res identity 14.61,
  Sinkhorn 14.59, diagonal 14.40; streams n=2 14.43, n=4 14.40, n=10 14.35. More loops (2 to 6) give diminishing
  returns; Hyperloop beats plain looping at every loop count. With fixed depth, a smaller block looped more
  loses: looped 2L→2L×6→2L 15.76 vs 2L→4L×3→2L 14.85. No ablation of the loop position embedding alone.
  Single runs (no seed variance reported).
- Cost: per loop with n streams and width C: W_pre, W_post, W_res each n×nC plus small biases/scalars, plus e_l
  (C). At C=16, n=2: about 3×64 + 16 + ~10 ≈ 220 per loop (about 1,750 for x8, 7.5%); n=4: about 800 per loop.
  These are my estimates from the formulas, not the paper's numbers.
- Fit: the strongest from-scratch evidence that a few loop-specific parameters close the gap to unique layers in
  LM perplexity. At hidden size 16 the relative cost is large; n=2 is the affordable form.

---

## 2. Cause 2 — the same function on every pass

Pattern: without a pass signal, every pass must apply one update rule. Papers report that step-conditioned
blocks can help in-distribution but hurt extrapolation / path independence (Geiping), and that the effect on LM
perplexity at fixed loop count is small or zero (SCSE Table 1).

### Dehghani et al. 2019 — Universal Transformers (read by extraction agent)
- Citation: M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, Ł. Kaiser. ICLR 2019. arXiv:1807.03819.
- Method: one block applied T times with tied weights, post-LN; a fixed sinusoidal position-and-timestep
  embedding P^t is added to the state at every step; optional per-position ACT halting.
- Causes: 2 (timestep embedding), 6 (ACT).
- Evidence: LAMBADA LM perplexity dev/test: Transformer 5122/7321; UT 6 steps fixed 279/319; UT 8 fixed 192/202;
  UT 9 fixed 214/239; UT dynamic halting 134/142. WMT14 En-De: UT base 28.9 vs Transformer base 28.0 (no ACT; ACT
  "marginally degraded results on MT"). No ablation of the timestep embedding.
- Cost: sinusoidal timestep embedding 0 params; ACT gate d+1 = 17 params.
- Fit: a loop embedding is a 0-parameter (sinusoidal) or r×16-parameter (learned) change. The UT paper gives no
  isolated evidence for it.

### Kim et al. 2026 — Looped Transformers with Source-Centered State Evolution (SCSE)
- Citation: B. J. Kim, K. Hayashi, S. Kamiya, M. Koyama, Y. Iwasawa, Y. Matsuo. arXiv:2607.27656.
  https://arxiv.org/abs/2607.27656
- Method: build an input-conditioned anchor h* = e + a_ω(e) once; evolve only the deviation Δ from the anchor
  with a zero-preserving shared block, Δ_{t+1} = Δ_t + m·s·G_θ(Δ_t), and read out h* + Δ_T. Its controls are
  the useful part for us: a "tuned adapter" with additive injection and a residual step scale s,
  h_{t+1} = h_t + s·B_θ(h_t + α·W_in·h*), and a "recurrent-step-conditioned" control that adds a projected
  sinusoidal loop-step embedding τ·P_step·γ_t to the block input.
- Causes: 2, 3, 4.
- Evidence: WikiText-103, loop depth sampled uniformly from 1..8 in training. Test PPL at T=8 (mean ± std):
  22M: tuned adapter 156.4 ± 0.6, step-conditioned 156.7 ± 0.5, SCSE 155.1 ± 0.6. 50M: plain looped baseline
  151.1 ± 0.9, step-conditioned 125.7 ± 0.4, SCSE 123.1 ± 0.2. 95.6M: plain looped 117.1 ± 0.5, tuned adapter
  98.6 ± 0.5, step-conditioned 100.8 ± 0.5, SCSE 96.9 ± 0.3. So the large gain over the plain looped model
  comes from the injection/anchor/step-scale package; the loop-step embedding on top of it adds nothing (22M) or
  hurts (95.6M). Best step-embedding scale τ = 0.015 (small).
- Cost: SCSE anchor module a_ω and initial-state module H_0 (sizes not extracted); step embedding r×d.
- Fit: evidence that loop-step embeddings alone are not where the gain is; evidence for input injection plus a
  residual step scale (sections 3 and 4).

### Geiping et al. 2025 — step-indexed core (negative result)
- From the Huginn paper (full entry in section 4): "We also evaluated s_i = R_i(e, s_{i−1}), i.e. a core block
  that takes the current step as input ... but find that this interacts badly with path independence, leading to
  models that cannot extrapolate." No numbers given.

(Further step-conditioning evidence from Xu & Sato 2025 and Xue et al. 2026 is in section 8; Hyperloop's loop
position embedding is in section 1 and has no separate ablation.)

---

## 3. Cause 3 — hidden-state drift, norm growth, instability, seed sensitivity

Pattern from these papers: (a) the residual-stream norm grows with each pass; (b) the loss curve can look normal
while the state explodes, because the final LayerNorm rescales h_T before the LM head (Buehlmaier 2026,
section 7.1); (c) instability appears as loss spikes or a high-loss plateau that worsens with loop count and
learning rate (Parcae Table 2; FLT Table 1; Ouro T=8); (d) plain looped models get *worse* with more loops
(FLT: small model, Wiki2 PPL 67.80 / 73.04 / 78.79 at 3 / 6 / 9 loops); (e) some initializations never
converge (Buehlmaier Table 4). Our x4 ≤ x2 and the 2.94/2.78/2.81 seed spread at x8 match (c)-(e). Cheap check:
log the RMS of the residual stream after each pass and the per-pass increment cosine.

### Prairie, Novack, Berg-Kirkpatrick, Fu 2026 — Parcae: Scaling Laws for Stable Looped Language Models
- Citation: H. Prairie, Z. Novack, T. Berg-Kirkpatrick, D. Y. Fu. arXiv:2604.12946 (ICLR 2026 per search
  listing; not verified in the PDF). https://arxiv.org/abs/2604.12946
- Method: writes the looped residual update as h_{t+1} = Ā·h_t + B̄·e + R(h_t, e). Prior looped models have
  Ā = I (additive injection, marginally stable) or an unconstrained Ā (concatenation adapter, can be unstable).
  Parcae sets Ā = exp(Δ·A) with A = Diag(−exp(log A)) (negative diagonal, so 0 < Ā < 1 per channel),
  B̄ = Δ·B (B unconstrained), a learned step Δ ∈ R^d, a LayerNorm on the injected prelude output e ("prelude
  norm"), a learned readout C before the coda, and per-sequence loop-count sampling.
- Causes: 3, 4, 5.
- Evidence: divergent pre-norm runs learn ρ(Ā) ≥ 1 and the recurrent state norm explodes (Fig. 2/3).
  LR sweep {2e-4..1e-3}: baseline pre-norm looped model converges only at 2e-4; residual-norm (sandwich)
  model at 2e-4 and 4e-4; Parcae at all five (Table 2). Table 4 (140M, retrofitted to a looped model): baseline
  diverges; +constrained A +per-sequence sampling: val loss 3.38/3.01/2.98 at T=1/4/8; +prelude norm
  3.28/2.97/2.95. Against Huginn-style RDMs: val PPL 14.23→13.59 (100M, T=16) and 10.76→10.09 (350M, T=8).
  Against parameter-matched Transformers (nanochat setup, middle third looped, T=8): val PPL 21.48→19.06 (140M),
  15.79→14.49 (370M), 13.08→12.49 (770M), 11.95→11.42 (1.3B). Core and Core-Extended have ± over seeds.
- Cost at our scale: log A 16 + Δ 16 + B 256 + C 256 + LayerNorm on e 32 = 576 parameters (2.5%). Compute:
  two 16×16 matmuls per pass.
- Fit: good. Directly targets norm growth and LR sensitivity, which is our lr 1e-2 regime. The exact input to R
  (whether R sees h_t or Ā·h_t + B̄·e) is written loosely in the paper (Algorithm 1 line 5: h_t ← Ā h_{t−1} +
  B̄ e + R(h_t, e)); the public code should be checked before implementation.

### Huang et al. 2026 — Towards Looped Models Done Right, Part II: Rethinking at Fixed Points (OrthoInj)
- Citation: B. Huang, C. Shi, J. Chen, S. Wen, Z. Liu, E. Xing, X. Ma. arXiv:2610.06833.
  https://arxiv.org/abs/2610.06833
- Method: orthogonal injection. With Parcae's diagonal carry Λ and q_t = Δ ⊙ W·e_t, the core input is
  ζ = Q_q·Λ·h + q with Q_q = I − q qᵀ/(‖q‖² + ε): the part of the carried state along the input is removed, so
  every pass receives the input at the same strength. Drops Parcae's prelude RMSNorm. Also a learned depth prior.
- Causes: 3, 4.
- Evidence: Table 3 (all trained with a log-normal-Poisson mean 5 depth prior, evaluated at R=5): S (100M, 21.5B
  tokens) val PPL DEQ-QKV 6.30, Huginn-Linear (concat adapter) 5.90, Parcae-Decay 5.75, OrthoInj 5.67; M (400M)
  Huginn-Linear 3.99, Parcae-Decay 3.96, OrthoInj 3.94; L (1.6B) Parcae-Decay 3.09, OrthoInj 3.08. Dropping the
  prelude RMSNorm and adding the projection each help and their gains add (Appendix C.2).
- Cost: same parameters as Parcae minus the prelude LayerNorm; one projection per token per pass.
- Fit: a refinement of Parcae; the gain over Parcae is 0.3-1.4% relative PPL. Second priority after a plain
  Parcae-style arm.

### Fu et al. 2026 — Simply Stabilizing the Loop via Fully Looped Transformer (FLT)
- Citation: R. Fu, Z. Yang, J. Zhang, J. Ma, H. Chen, Y. Li, Y. Chang. arXiv:2605.18797.
  https://arxiv.org/abs/2605.18797
- Method: two parameter-free changes. (1) Fully Looped Architecture: the previous loop's output is made available
  to every layer of the current loop, not only the first. (2) Attention Injection: for loops t > 1, each attention
  block uses the previous loop's output h_L^(t−1) as the query and the current stream as keys and values, reusing
  W_Q, W_K, W_V; in the first layer the input embedding x provides keys and values.
- Causes: 3, 4, 5. The paper names two patterns: early gradient oscillation and residual explosion.
- Evidence: FineWeb-Edu, 20 tokens/parameter, µP, Muon + AdamW. Table 1 (Wiki2 PPL / downstream average):
  small (127M, 6 layers): plain LT 67.80/32.57 (3 loops), 73.04/31.47 (6), 78.79/30.22 (9); LT with input
  injection best at 6 loops (average 33.35); FLT 32.13→32.45 from 3 to 9 loops. Base (318M, 12 layers): plain LT
  40.52 avg at 3 loops, 36.56 at 6, collapses at 9; FLT 39.90/40.47 PPL at 3 → 41.72/38.44 at 9. At 12 loops all
  base models except FLT collapse (Fig. 4). Plain addition of the previous loop to every layer (FLT_res) still
  collapses.
- Cost: 0 parameters; changes the attention wiring.
- Fit: moderate. At small scale, plain input injection was as good as FLT; the gain is at higher loop counts and
  larger models. More invasive than injection.

### Wang et al. 2026 — On the Residual Scaling of Looped Transformers
- Citation: S. Wang, B. Li, G. Zhang, W. Huang, S. Yan, J. Li. arXiv:2606.18524 (v2, Sep 2026).
  https://arxiv.org/abs/2606.18524
- Method: scale each residual branch by ε, h ← h + ε·f(h). Shared weights make the per-pass updates correlated,
  so the sum grows as Θ(N) and needs ε = 1/N (not 1/√N); for L unique layers looped N times,
  ε = λ/(N·√L). Consequence: the optimal learning rate depends on L only, not on N.
- Causes: 3, 5.
- Evidence: L=12 pre-norm Llama-style looped models, FineWeb-Edu 10B tokens, N ∈ {1,2,4,8}, AdamW: at N=8,
  1/N scaling gives a 0.025-nat lower minimum loss than 1/√N; the optimal LR stays nearly constant across N
  under 1/N scaling. After training, per-pass increments stay positively correlated (cosine 0.3-0.6 for early
  and middle passes). Single seed per configuration (stated limitation).
- Cost: 0 parameters. v2 has no learned scalars: λ = 1, fixed (Table 4).
- Used (read in full, v2): Table 3 implements x + λ N⁻¹ m_L^{-1/2}·f(LN(x)) on both the attention and the MLP
  branch, m_L = L/12 (their tuned reference depth), so ε = 1/N at L = 12; the headline ε = λ/(N√L) differs from
  Table 3 by √12. Looped-block LR and AdamW ε also scale by m_L^{-1/2}. Embeddings, final norm and head are
  outside the loop and unscaled. Subset loops (middle layers only) are not covered. In recursion_comparison the
  LR was set on the same model whose middle pair is looped, so m_L = 1 and ε = 1/passes; LR unchanged.
- Fit: very good and very cheap. Our LR (1e-2) was set for the 4-layer model; without loop-aware scaling the
  effective update of the shared layers grows with r (same argument as DeepLoop below), which matches the x8
  seed spread. Note that the paper loops the whole stack; our loop is the middle pair, so N = r for layers 2-3
  only.

### Li et al. 2026 — DeepLoop: Depth Scaling for Looped Transformers
- Citation: S. Li, Y. Zhang, J. Guo, Q. Gu, M. Wang. arXiv:2607.13491 (v2). https://arxiv.org/abs/2607.13491
- Method: a shared block aggregates gradients from all its visits, so DeepNorm's exponent must rise with loop
  count. Post-LN DeepNorm block with a normalized branch input, residual multiplier α = (2N)^{1/2} and init scale
  β = (8N)^{−1/2} for unrolled depth N.
- Causes: 3, 5.
- Evidence: Table 1, FineWeb-Edu 50B tokens, final val loss, loop count R=1/3/5/7. GPT-2 small: pre-LN baseline
  2.8627/2.8077/2.7910/2.7700, DeepLoop 2.8631/2.7917/2.7679/2.7514. GPT-2 medium: baseline
  2.6253/2.5779/2.5640/2.5558, DeepLoop 2.6264/2.5627/2.5444/2.5280. Single seed.
- Cost: 0 parameters; changes norm placement and init.
- Fit: moderate; requires switching the block to post-LN sandwich, a larger change than residual scaling.

### Marchenko et al. 2026 — Closing the Loop: Practical Training Recipes for Looped Language Models
- Citation: A. Marchenko, V. Bezrukov, O. Kashurin, I. Fedorova, D. Bocharov, Y. Shakhvalieva, M. Tikhonova,
  V. Ternovskii. arXiv:2610.00673 (under review at ICLR 2027). https://arxiv.org/abs/2610.00673
- Method / findings: (1) normalization ablation for a 24-layer shared block, T=4: "Ouro-norm" (normalize input
  and output of each sublayer, i.e. Peri-LN) and pre-norm train; post-norm and sandwich-norm fail because
  gradients vanish through the block. (2) A final norm before the shared LM head is needed for stability.
  (3) Dense-to-loop conversion: looping Qwen3-1.7B without input re-injection makes later steps worse (steps 2
  and 3 end 0.01 and 0.02 nats above step 1); one learned scalar z_t = w·e + (1−w)·h_{t−1}, w = σ(a), init
  w = 0.99, makes steps 2 and 3 end about 0.02 nats below step 1. Step LoRA rank 32 gives little benefit.
  (4) An Ouro T=8 run: loops 5-8 change the loss by about 0.01 nats or less, so they use T=4.
- Causes: 3, 4, 5.
- Evidence: Table 5 (LM loss after 19K steps): Ouro-norm + RMS final 1.98; pre-norm + RMS 2.03; pre-norm
  + parameter-free RMS 2.07; pre-norm without final norm 2.11; post-norm 7.37; sandwich 7.40. Table 6: without a
  final norm 5-11 loss spikes and max gradient norm up to 5.2e7; with one, 0 spikes and max 15.7-42.3. Post-norm
  at T=1-3 converges to 2.66-2.82, at T=4 returns to the ~7.4 plateau.
- Cost: scalar injection 1 parameter; Peri-LN 64 parameters per layer.
- Fit: very good. Two cheap arms: scalar-gated input injection (1 parameter) and Peri-LN on the shared pair (128
  parameters). Our model already has a final LayerNorm (ln_f).

### Kim et al. 2025 — Peri-LN: Revisiting Normalization Layer in the Transformer Architecture
- Citation: J. Kim, B. Lee, C. Park, Y. Oh, B. Kim, T. Yoo, S. Shin, D. Han, J. Shin, K. M. Yoo. ICML 2025.
  arXiv:2502.02732. https://arxiv.org/abs/2502.02732
- Method: normalization around each sublayer: x ← x + Norm(F(Norm(x))) (input and output normalized; residual
  stream not normalized).
- Cause: 3 (activation growth, seed sensitivity). Not loop-specific.
- Evidence: 5 seeds per setting, divergences excluded. 400M: average benchmark Pre-LN 49.69 vs Peri-LN 51.57,
  loss 3.43 vs 3.34; per-task std clearly smaller with Peri-LN (e.g. ARC-Easy ±1.63 vs ±0.81, Winogrande ±2.35 vs
  ±0.67). Pre-LN shows divergences and loss/gradient spikes across seeds (Fig. 4).
- Fit: good; the seed-variance reduction is the pattern we need at x8. 64 extra parameters per layer.
- Used (read in full, v3): Eq. 3, x ← x + Norm(F(Norm(x))); output-norm gain learnable, init 1 (Sec. 5.3,
  App. L); RMSNorm ≈ LayerNorm (App. G.4), so LayerNorm with bias here; the embedding norm is optional (App. G.5)
  and not used, because only the looped pair gets Peri-LN in recursion_comparison.

### Labovich 2026 — Stability and Generalization in Looped Transformers
- Citation: A. Labovich. arXiv:2604.15259 (v2). https://arxiv.org/abs/2604.15259
- Method / theory: looped networks without recall (input injection) have countable fixed points and only select
  among them ("basin selection"); recall plus outer normalization gives fixed points that are reachable, depend
  smoothly on the input, and have stable backpropagation. Introduces internal recall (the input enters only the
  sublayer update, not the residual): x_{t+1} = φ2(z + h2(g(z, x0))), z = φ1(x_t + h1(g(x_t, x0))).
- Causes: 3, 4.
- Evidence: single-layer looped transformers on chess, sudoku, prefix sums. Best normalized model is 12 and 2
  points better on hard data than the best un-normalized one on sudoku and chess; recall models beat autonomous
  models on hard data across all norm/LR pairs; at high LR, ρ(W_x) grows and models lose usable fixed points.
- Fit: theory support for injection plus normalization; no LM evidence.

### Buehlmaier 2026 — LayerNorm as Implicit Gain Control in Looped Transformers
- Citation: M. M. M. Buehlmaier. arXiv:2607.10681. https://arxiv.org/abs/2607.10681
- Findings: in pre-LN looped blocks, the in-block LayerNorm couples the block's Lipschitz constant inversely to
  activation scale. Parcae's carry constraint ρ(Ā) < 1 is necessary but not sufficient; a minority of
  initializations never converge to a fixed point. Toy reproduction of Parcae Table 2 (d=32): an unconstrained
  carry learns ρ = 1.20 at lr 3e-2 and 1.84 at lr 1e-1 with state norm about 9,570, while the loss stays finite
  "because the Coda's LayerNorm normalizes h_T before the unembed — loss curves alone can hide this failure
  mode". The learned carry serves stabilization, not memory, in cross-channel tasks.
- Fit: diagnostic guidance (log state norms per pass; expect seed-dependent non-convergence); CPU-scale only.

(MoEUT's peri-layernorm and LT2's attention output gate, both cause 3, are in section 6/9.)

---

## 4. Cause 4 — the input is forgotten across passes (input injection / recall)

Pattern: without injection, later passes can make the representation worse (Closing the Loop, Qwen conversion:
+0.01/+0.02 nats at steps 2/3); performance falls when iterating beyond the training count ("overthinking",
Bansal); the fixed point does not depend smoothly on the input (Labovich). With injection, curves stay flat or
improve with more passes.

### Bansal et al. 2022 — End-to-end Algorithm Synthesis with Recurrent Networks
- Citation: A. Bansal, A. Schwarzschild, E. Borgnia, Z. Emam, F. Huang, M. Goldblum, T. Goldstein. NeurIPS 2022.
  arXiv:2202.05826. https://arxiv.org/abs/2202.05826
- Method: (1) recall: concatenate the input x to the features before each recurrent block, r([φ_{k−1}, x]), with
  one conv layer mapping back to the feature width. (2) Progressive loss: sample n ~ U{0, m−1}, k ~ U{1, m−n};
  run n iterations without gradient, k more with gradient; L = (1−α)·L_max_iters + α·L_progressive.
- Causes: 4, 5, 7.
- Evidence: prefix sums (train 32 bits, m=30, 400 filters, α=1): without recall/progressive loss 0% on 512-bit;
  DT-Recall + progressive > 97% after ~200 iterations; recall without progressive loss reaches the same accuracy
  but needs ~2× the iterations; n=0 instead of random n: 90.27% vs 97.12%. Mazes (train 9×9, test 59×59, α=0.01):
  recall + progressive 97% vs recall alone 83%; without recall all models fail. Chess: +4% accuracy; non-recall
  models decay after ~70 iterations. "Only models with recall avoid the overthinking trap."
- Cost: one extra input map (2h→h).
- Fit: injection fits (section 10). Progressive loss needs extra forward passes and varying iteration counts; it
  mainly serves extrapolation (cause 7), which our fixed-count protocol does not test.

### Geiping et al. 2025 — Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn)
- Citation: J. Geiping, S. McLeish, N. Jain, J. Kirchenbauer, S. Singh, B. R. Bartoldson, B. Kailkhura,
  A. Bhatele, T. Goldstein. arXiv:2502.05171. https://arxiv.org/abs/2502.05171
- Method: prelude P (embeds the input), recurrent block R iterated r times, coda C. Each step
  s_i = R(e, s_{i−1}) re-injects the prelude output e through an adapter A: R^{2h}→R^h on [s_{i−1}; e]; s_0 is
  random noise. Sandwich RMSNorm layout; embedding scaled by √h. r sampled from a log-normal Poisson (mean 32)
  per step; truncated backprop through the last k=8 iterations.
- Causes: 3, 4, 5, 7.
- Evidence: "re-incorporation of initial embedding features via addition rather than concatenation works
  equally well for smaller models, we find that concatenation works best at scale." "At small scales, most
  normalization strategies, e.g. pre-norm, post-norm and others, work almost equally well." At 3.5B scale: bad
  run 1 (parameter-free RMSNorm, no embedding scale, additive adapter, lr 4e-4) collapsed: token correlation of
  hidden states → 1; bad run 2 learned to ignore the state (same PPL at 1 and 32 recurrences); fixed by the
  sandwich block and lr 4e-5. Truncation at k=8 "works as well as sampling k uniformly" at small scale. Noise
  injection between iterations did not help. Step-indexed core did not extrapolate.
- Cost: concat adapter 2h×h = 512 (+16 bias) parameters at h=16; additive injection 0.
- Fit: the paper states additive injection is as good as concatenation at small scale, which favors the
  0-parameter additive form for our 23k-parameter model.

### Fu et al. 2026 (FLT), Huang et al. 2026 (OrthoInj), Prairie et al. 2026 (Parcae), Marchenko et al. 2026
(scalar injection), Nguyen & Lin 2025 (additive x = h + e) — see section 3 and 0. All of them inject the
input at each pass. FLT small scale (127M): the text states that plain LT degrades with loops (average
32.57 → 30.22 from 3 to 9 loops) while "LTi and FLT are more stable, with LTi obtaining the best small-size
average of 33.35 at 6 loops" (the per-row alignment of the extracted Wiki2 PPL column is garbled because
collapsed runs are blank, so I quote the text only).

---

## 5. Cause 5 — optimization through many shared passes

Pattern: loss spikes and gradient oscillation early in training that grow with loop count (FLT Fig. 2; Ouro
8→4 loops), gradient norm concentrated in one layer with post-norm (Closing the Loop Fig. 6), and an effective
learning rate on shared weights that grows with the number of visits (DeepLoop; residual-scaling paper).

Evidence summary (entries above): residual scaling 1/N (Wang et al.), DeepLoop scaling, Parcae's stable carry,
FLT attention injection, truncated backprop (Geiping, k=8 "works as well" at small scale; Parcae µ_bwd =
⌈µ_rec/2⌉), Muon vs AdamW (McLeish below), progressive loss (Bansal), deep supervision and 1-step gradients
(HRM/TRM, section 8).

### McLeish et al. 2025 — Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence
- Citation: S. McLeish, A. Li, J. Kirchenbauer, D. S. Kalra, B. R. Bartoldson, B. Kailkhura, A. Schwarzschild,
  J. Geiping, T. Goldstein, M. Goldblum. NeurIPS 2025. arXiv:2511.07384. https://arxiv.org/abs/2511.07384
- Method: convert pretrained Llama/TinyLlama/OLMo layers into prelude/recurrent/coda with a concat adapter;
  schedule the mean recurrence linearly upward during training (a depth curriculum).
- Causes: 5, 1 (pretrained init).
- Evidence: pretrained init beats random init; linear recurrence curriculum has "a small positive impact" on
  validation loss per step and saves FLOPs. Muon beats AdamW for recurrent models: "the AdamW line ends early as
  the loss spikes and becomes NaN" (TinyLlama (4,8,4), lr 5e-5 AdamW vs 1e-3 Muon); for the non-recurrent model
  the difference is small.
- Fit: the curriculum changes the training protocol (loop count varies during training). Muon is excluded from
  this proxy by the user's choice (AdamW only); the evidence is recorded, not proposed.

---

## 6. Cause 6 — the same compute for every token (adaptive depth)

Pattern: halting gates collapse (to one step, or to the last step) unless regularized; ACT is sensitive to its
ponder weight; adaptive methods gain mostly in compute, not in perplexity, at small scale.

### Bae et al. 2025 — Mixture-of-Recursions (router part)
- Method: Middle-Cycle shared stack; a lightweight linear router picks, per recursion step, which tokens continue
  (expert-choice, top-k by capacity schedule) or assigns each token a depth once (token-choice); KV cache per
  recursion or shared.
- Evidence (360M class, equal training FLOPs): vanilla 315M NLL 2.7824, few-shot 42.3; recursive Nr=2 167M 2.8079,
  42.6; MoR expert-choice Nr=2 167M 2.7511, 43.1; MoR expert Nr=3 118M 2.7925; MoR token-choice Nr=3 2.9163, 40.0.
  "At 135M, MoR trails vanilla; the gap closes at larger scales." The MoR gain at equal FLOPs comes from
  processing more tokens per FLOP.
- Cost: router Linear(d,1) per recursion (17 params each at d=16) plus auxiliary loss.
- Fit: poor for this proxy: our comparison fixes tokens and compute per arm; MoR trails vanilla at its smallest
  scale.

### Universal Transformer ACT (above), PonderNet, SUT, Ouro, MoEUT — read by extraction agent

- **PonderNet** (A. Banino, J. Balaguer, C. Blundell, ICML 2021 AutoML workshop, arXiv:2107.05407): halting
  probabilities λ_n, p_n = λ_n Π_{j<n}(1−λ_j); loss Σ p_n·L(y, ŷ_n) + β·KL(p ‖ geometric prior λ_p). No LM
  experiment. bAbI: Transformer + PonderNet 0.15 ± 0.9 average error vs UT 0.29 ± 1.4; parity extrapolation where
  ACT fails. Fit: none for a fixed loop count.
- **Sparse Universal Transformer** (S. Tan, Y. Shen, Z. Chen, A. Courville, C. Gan, arXiv:2310.07096): MoE
  attention and FFN plus stick-breaking halting. WMT14 En-De: UT 28.9, SUT base 29.2 BLEU. Ablation (valid
  loss/BLEU): full 2.192/29.2, minus halting 2.219/29.1. MoEUT's comparison on C4 [checked]: SUT 40.50 PPL, SUT
  without ACT 21.51, dense baseline 18.97 (44M), i.e. ACT hurt LM perplexity badly. Fit: none.
- **Ouro / LoopLM** (R.-J. Zhu et al., arXiv:2510.25741): shared stack reused up to T_max = 4, LM head after each
  loop, exit gate λ_t = σ(Linear(h^(t))), loss Σ_t p(t|x)·L^(t) − β·H(p) (entropy = KL to a uniform prior).
  "Sandwich normalization ... an RMSNorm layer before both the attention and FFN sub-layers" [checked wording].
  Stability: 8 recurrent steps gave "loss spikes and gradient oscillations", fixed by reducing to 4 steps, larger
  batch, β 0.1 → 0.05. Appendix D (53M-1.36B, 20B tokens, loops 1/2/4/8): a non-shared model of equal depth beats
  LoopLM, "this gap increases with the recurrent step and decreases with the model size". Performance peaks at the
  trained depth T=4 and degrades for T=5-8. Uniform prior beats geometric priors (776M, figure only).
  Fit: the per-step loss changes the training objective; the gate is useless at fixed r. Not proposed.
  ("Closing the Loop" adds: gate collapse is invisible in the LM loss; warmup and β = 0.15 prevent it.)
- **MoEUT** (R. Csordás, K. Irie, J. Schmidhuber, C. Potts, C. D. Manning, NeurIPS 2024, arXiv:2405.16039):
  σ-MoE FFN and SwitchHead attention in a shared stack, plus two general parts: (a) layer grouping: loop a group
  of G different layers (ABAB, not AABB), G=2 best below 300M; (b) "peri-layernorm" in their sense: no
  LayerNorm on the main data path; LN only before the query/key projections, the expert-selection gates and the
  final classifier, because with pre-LN the residual norm grows and shared layers cannot rescale their output
  per depth. C4 PPL [checked]: 44M dense 18.97 vs MoEUT 18.30; 244M 13.40 vs 13.24; 1040M 11.15 vs 10.90. Their
  peri-LN beats pre-LN and post-LN at 44M and 244M (figure only; gap larger for small models). Fit: grouping =
  our current ABAB design; MoE not applicable at d=16; the LN-placement idea is 0 extra parameters but untested
  below 44M and with GELU MLPs with biases.

---

## 7. Cause 7 — train/test loop-count mismatch

Not active in our protocol (fixed r in training and evaluation). Evidence recorded for completeness:
Bansal (progressive loss; recall prevents overthinking), Geiping (random r and truncated backprop; PPL improves
up to 64 test recurrences), Parcae (per-sequence depth sampling; test-time looping follows a saturating
exponential), SCSE (PPL rises beyond the trained range for every method on WikiText-103), Ouro (degrades beyond
T=4), Neural Data Router (train 20 layers, test 24 without penalty).

### Csordás, Irie, Schmidhuber 2022 — The Neural Data Router (read by extraction agent)
- Citation: ICLR 2022, arXiv:2110.07732.
- Method: copy gate on a shared post-LN Transformer: h_new = g·u + (1−g)·h, g = σ(FFN_gate(a)), gate bias
  initialized to −3 (start near identity); geometric attention.
- Causes: 5, 2/4, 7.
- Evidence: compositional table lookup (longer split, forward/backward): Transformer 0.13/0.12; NDR 1.00/1.00;
  ListOps depth 7-8: Transformer 0.74, NDR 0.99. No language modeling.
- Cost: gate 2d²+2d = 544 parameters per shared layer at d=16 (or 17 for a scalar gate, untested).
- Fit: outside its evidence domain (algorithmic seq2seq).

### Csordás, Irie, Schmidhuber 2021 — The Devil is in the Detail (read by extraction agent)
- Citation: EMNLP 2021, arXiv:2108.12284.
- Finding: a plain weight-shared Transformer "without adaptive computation time and timestep embedding" is a
  strong baseline on systematic generalization; embedding/positional scaling (PED) and no early stopping matter;
  "reducing the number of parameters (either using Universal Transformers or reducing the state size) usually
  stabilizes the network". No LM numbers.
- Fit: supports plain sharing as a valid baseline; no method to add.

---

## 8. Fixed-point, implicit-layer and theory papers (read by extraction agent)

### Bai, Kolter, Koltun 2019 — Deep Equilibrium Models
- Citation: NeurIPS 2019, arXiv:1909.01377. https://arxiv.org/abs/1909.01377
- Method: weight-tied layer with input injection z_{i+1} = f(z_i; x), solved for the fixed point by Broyden
  root-finding and trained by implicit differentiation (constant memory). Transformer form: x̃ = x·W_x added to
  the Q/K/V input; post-LN.
- Causes: 3, 4.
- Evidence: WikiText-103, small (4.5M non-embedding): Transformer-XL 4 layers 35.8 PPL; weight-tied 16 layers 34.9;
  DEQ 32.4 (memory 4.8 / 6.8 / 1.1 GB). Medium: 16-layer 24.3 vs DEQ 24.2. Plain weight-tied stacks oscillate
  around the fixed point instead of converging (Appendix D). DEQ is 1.6-2.8x slower.
- Fit: the solver does not fit a fixed-r proxy; the input-injection form (W_x, 272 params, or additive) does.

### Bai, Koltun, Kolter 2021 — Stabilizing Equilibrium Models by Jacobian Regularization
- Citation: ICML 2021, arXiv:2106.14342. https://arxiv.org/abs/2106.14342
- Method: penalty γ·‖εᵀJ_f(z*)‖²/d with ε ~ N(0, I) (Hutchinson estimate of ‖J_f‖_F²), applied with probability
  p per step, γ ramped up.
- Causes: 3, 5.
- Evidence: WikiText-103 98M DEQ-Transformer: pre-LN DEQ diverges; post-LN 24.0 PPL at 30 function evaluations;
  pre-LN + JR 24.5 at 14; post-LN + JR 24.9 at 12. Weight decay did not stabilize. The authors say JR "does not
  fundamentally solve the growing instability problem, but only empirically alleviates it".
- Cost: 0 params; one extra vector-Jacobian product (~+30% memory) on regularized steps.
- Fit: weak; evidence is for equilibrium solvers, and applying it at the r-th iterate of a fixed unroll is
  untested. γ values do not transfer.

### Anil et al. 2022 — Path Independent Equilibrium Models Can Better Exploit Test-Time Computation
- Citation: C. Anil, A. Pokle, K. Liang, J. Treutlein, Y. Wu, S. Bai, J. Z. Kolter, R. Grosse. NeurIPS 2022.
  arXiv:2211.09961. https://arxiv.org/abs/2211.09961
- Method: path independence (same limit from any initial state), measured by the Asymptotic Alignment score;
  raised by mixed initialization (zero or Gaussian initial state) and randomized iteration counts; input
  injection and weight tying are argued necessary.
- Causes: 7, 4, 3.
- Evidence: prefix sums and mazes (conv ResNets): non-path-independent net AA 0.62, 66.66% on 64-bit prefix sums;
  path-independent net AA 0.99, 100%. Removing input injection loses upward generalization on mazes (figure only).
- Fit: the interventions change the fixed loop-count protocol (random depth) or need a free initial state, which
  our model does not have. Supports input injection.

### Schwarzschild et al. 2021 — Can You Learn an Algorithm?
- Citation: A. Schwarzschild, E. Borgnia, A. Gupta, F. Huang, U. Vishkin, M. Goldblum, T. Goldstein. NeurIPS
  2021. arXiv:2106.04537.
- Finding: plain weight-shared ResNets (no injection, no norm) beat same-depth unique nets on easy-to-hard
  algorithmic tasks (e.g. mazes 13×13, depth 44: 29.72% vs 22.53%). Prefix-sum training was unstable (only runs
  that reach 100% train accuracy kept). No LM data; opposite direction to LM results.

### Wang et al. 2025 — Hierarchical Reasoning Model (HRM)
- Citation: G. Wang, J. Li, Y. Sun, X. Chen, C. Liu, Y. Wu, M. Lu, S. Song, Y. Abbasi Yadkori. arXiv:2506.21734.
- Method: low-level module updated every step, high-level module every T steps, inputs combined by addition,
  1-step gradient (only the last call carries gradient), deep supervision over segments with detached state,
  Q-learning halting; post-norm RMSNorm.
- Causes: 5, 3. Evidence: 27M params, ~1000 examples: Sudoku-Extreme 55.0%, Maze-Hard 74.5%, ARC-AGI-1 40.3%.
  No ablation table.
- Fit: none for next-token LM with a fixed loop count.

### Jolicoeur-Martineau 2025 — Less is More: Recursive Reasoning with Tiny Networks (TRM)
- Citation: A. Jolicoeur-Martineau. arXiv:2510.04871.
- Method: one 2-layer net used recursively on (x, y, z) with addition; full backprop through the last recursion,
  deep supervision up to 16 steps, EMA of weights 0.999.
- Evidence (Sudoku-Extreme, 1000 examples): TRM 87.4%; with 1-step gradient 56.5%; no EMA 79.9%; 4 layers 79.5%;
  separate f_H, f_L 82.4%. "On small data HRM tends to overfit quickly and then diverge"; EMA "prevents sharp
  collapse".
- Fit: only EMA of weights transfers (0 params), and it is an evaluation/training-protocol change for every arm,
  not a looped-arm method. The 1-step-gradient ablation is evidence against truncating gradients in a small
  model.

### Xu & Sato 2025 — On Expressive Power of Looped Transformers: Timestep Encoding
- Citation: K. Xu, I. Sato. ICML 2025. arXiv:2410.01405. https://arxiv.org/abs/2410.01405
- Method: theory: weight-tied FFNs need a per-loop scaling η(t) to remove approximation error. Practical model:
  AdaLN-zero-style modulation from a sinusoidal timestep embedding: X' = X + γ1(t)·Attn(α1(t)·RMSN(X)),
  out = X' + γ2(t)·FF(α2(t)·RMSN(X')), γ initialized to 0 and α to 1. Input embeddings are added at every loop
  in all experiments.
- Causes: 2 (and robustness).
- Evidence [checked]: WikiText-103 (d=256 class models, 100k iterations): standard TF train/test PPL 15.9/20.5;
  looped r=24 17.1/20.6; looped + timestep encoding r=24 15.9/19.6. In-context decision trees MSE: TF 8.6e-3,
  looped 1.4e-2, + timestep 1.7e-3. Reasoning tasks: gains mostly at large r (edit distance 60, r=100: 47.7 →
  88.3). No seed variance.
- Cost: the MLP generator costs about 1,632 params at d=16 (7%); a per-pass table of (γ1, γ2) per layer costs
  2×16 per layer per pass, i.e. 64/128/256 params for x2/x4/x8 on the looped pair (α tables would double it).
- Fit: good for cause 2; the best LM evidence for pass-specific parameters that are cheap at our width. Tested
  on a whole-network 1-layer loop, not a middle pair.

### Xue et al. 2026 — Beyond What Seems Necessary (loop embedding as a side result)
- Citation: Y. Xue, ..., B. Mirzasoleiman. arXiv:2602.00927.
- Finding: a learned loop-index embedding added before the shared block did not change the trend of OOD accuracy
  vs training loops on a synthetic induction task (8 seeds); no with/without numbers. No evidence for or
  against loop embeddings in LMs.

### Kohli et al. 2026 — Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers
- Citation: H. Kohli, S. Parthasarathy, H. Sun, Y. Yao (as extracted). COLM 2026. arXiv:2604.07822.
- Method: plain looped transformer without injection; zero-initialized output projections of attention and FFN
  in every recurrent block (block starts as identity); fixed or Poisson-sampled loop count.
- Causes: 3, 5, 7.
- Evidence (synthetic multi-hop knowledge task): with default Gaussian init of the output projections "results
  are mixed across training runs and increasing recurrence does not yield a clear or consistent pattern"; R=5
  over 5 seeds gave three different behaviours; with zero-init the trends are consistent across seeds.
- Cost: 0 params.
- Fit: the only ablation in the corpus that targets seed-to-seed inconsistency of a looped model by
  initialization. Zero-init of the looped pair's attention out_proj and mlp c_proj is a 0-parameter change.
  Evidence is synthetic, not LM.

### Yang, Lee, Nowak, Papailiopoulos 2024 — Looped Transformers are Better at Learning Learning Algorithms
- Citation: L. Yang, K. Lee, R. Nowak, D. Papailiopoulos. ICLR 2024. arXiv:2311.12424.
- Method: Y_{t+1} = M(Y_t + P) with the prompt embedding P added every loop (additive injection); loss on every
  loop inside a window (truncated BPTT).
- Evidence: in-context linear regression, 1 looped layer (0.79M) matches a 12-layer GPT-2 (9.48M); without
  injection the test error rises past the trained loop count, with injection it stays low to 100 loops (figure).
  "A larger T could induce instability in training."
- Fit: supports 0-parameter additive injection.

### Fan, Du, Ramchandran, Lee 2025 — Looped Transformers for Length Generalization
- Citation: Y. Fan, Y. Du, K. Ramchandran, K. Lee. ICLR 2025. arXiv:2409.15647.
- Finding: input embeddings added to the previous step's output each loop; the no-injection ablation is
  generally worse (figure only). Needs per-sample step counts; not an LM method.

### Lan et al. 2020 — ALBERT
- Citation: Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, R. Soricut. ICLR 2020. arXiv:1909.11942.
- Finding (what to share, ALBERT-base, E=768 / E=128 averages): not-shared 82.3 / 81.6; shared-attention (FFN
  unique) 81.6 / 81.7; shared-FFN 79.5 / 80.2; all-shared 79.8 / 80.1. "Most of the drop comes from sharing FFN
  parameters." Hidden states oscillate rather than converge across layers.
- Fit: a per-pass unique FFN adds 2,128 params per layer per extra pass, far beyond the budget. Recorded as
  evidence that the FFN is where sharing costs most.

### Dabre & Fujita 2018 — Recurrent Stacking of Layers for Compact NMT
- Citation: R. Dabre, A. Fujita. arXiv:1807.05353.
- Finding: one shared layer stacked 1-6 times; unique 6 layers win in every language pair (e.g. Tr-En 17.31 vs
  18.36 BLEU); gains per extra recurrence shrink with depth; the lowest-resource pair is non-monotonic (7.99 at 3
  vs 7.91 at 4), like our x2 vs x4. Decoding with a different recurrence count than trained hurts sharply.
- Fit: baseline evidence only.

### Hegazy, Alanwar, Elhoushi 2026 — Gated Recurrent Transformers (GRT)
- Citation: A. Hegazy, A. Alanwar, M. Elhoushi. "Gated Recurrent Transformers: Expressive Depth through Recurrent
  Modulation". arXiv:2608.15062v4 (26 Aug 2026), preprint. Verified to exist; read from the PDF text.
- Method: prelude → shared core ×R → coda. Each step: h̃ = W_proj[h^(r−1) + ε_x ; h_pre] (prelude re-injection by
  concatenation and projection), o = B(h̃), gate g = σ(f_g([LN(h^(r−1)); LN(h_pre)])/τ + ε_g) with f_g a 2-layer
  SiLU MLP whose output bias is initialized to +4, update h^(r) = g ⊙ h^(r−1) + (1−g) ⊙ o. State noise
  ε_x ~ N(0, 0.1²) and gate noise in training only; R sampled uniformly per step.
- Causes: 2, 4, 3, 7.
- Evidence [checked]: nanoGPT on ~9.8B tokens, AdamW lr 6e-4. Component ablation, small model (dense 12 layers
  3.157): recurrence only 3.264; + prelude/coda 3.229; + state noise 3.211; + prelude re-injection 3.189; + gate
  3.141. Medium, 2,000 steps: recurrence 4.163; + prelude/coda 4.118; + noise 4.099; + re-injection 3.901 (−0.198);
  + gate 3.786 (−0.115). Seeds (small, 3 seeds): dense 3.188 ± 0.056, GRT 3.145 ± 0.004. Gate bias +4/+2/0/−2:
  3.141/3.152/3.160/3.151. IsoFLOP small: dense 3.15, GRT (35M) 3.14, RRT 3.14, Ouro-style 3.19, MoR 3.30.
- Cost at d=16: W_proj 512 + gate MLP ~800 + two LayerNorms 64 ≈ 1,376 params (6%); FLOPs +10d² per step.
- Fit: the closest published ablation to our question (component by component on a looped LM). Re-injection and
  the gate are the two largest components after the prelude/coda split (which our arms already have). The full
  GRT exceeds 1% of our budget; cheaper forms (additive re-injection, 0 params; a single-linear gate
  Linear(32→16) = 528 params) are untested reductions.

### Jaggi 2026 — Tying the Loop: Tied Expert Layers in MoE Language Models
- Citation: M. Jaggi. arXiv:2606.16825.
- Findings: unique 2+2 prelude/coda is the largest single effect; untying attention helps by 0.030-0.058; for
  tied weights, scaling the LR by 1/√g (loss 3.503) or 1/g (3.506) beats no scaling (3.542), tested with Muon.
- Fit: confirms our unique first/last layers; LR scaling is evidence for loop-aware scaling (with Muon, not
  AdamW).

---

## 9. Other mechanisms found

### Deng et al. 2026 — LT2: Linear-Time Looped Transformers (read by extraction agent)
- Citation: C. Deng, Y. Zhang, R.-J. Zhu, Y. Xu, J. Liu, T. S. E. Ng, H. Chen. arXiv:2605.20670 (v2).
- Method: looped stacks with linear/sparse token mixers; a zero-initialized per-channel residual across loops
  h^(τ) = h̃^(τ) + ρ_τ ⊙ h^(τ−1) [checked]; a head-specific sigmoid gate after attention (from Gated Attention).
- Causes: 3, 6.
- Evidence: 1.3B, T=4, 100B FineWeb-Edu tokens. Attention output gate (Table 4) [checked]: looped Transformer
  PPL 9.87 → 9.39, average 59.27 → 60.70. Residual-stream RMS grows within and across loops, "reaching ~20 times
  by the end of loop 4"; the attention sink is re-injected every loop; the gate "substantially mitigates — though
  does not eliminate" this. ρ_τ has no ablation. Single seed.
- Cost: ρ_τ = 16 params per pass; output gate about 16×4 + 4 = 68 per shared layer if it is a per-head scalar
  from the hidden state (exact form in the Gated Attention paper, not read here).
- Fit: the attention output gate is a cheap, well-evidenced (at 1.3B) stability fix for norm growth in loops.

---

## 10. Ranked shortlist of proxy arms

Ranking criteria: (1) evidence on language modeling with a looped model trained from scratch, (2) the method
targets the patterns we see (x4 ≤ x2, x8 seed spread 2.94/2.78/2.81, large gap to unique layers), (3) extra
parameters ≤ about 2.5% of 23,200, (4) the protocol stays fixed (same data, AdamW, lr, schedule, fixed r in
training and evaluation, 3 seeds). All arms below change only the architecture or the initialization.
"e" means the prelude output, i.e. the hidden state after layer 1 (layer 1 is the prelude, layers 2-3 the core,
layer 4 plus ln_f the coda), as in Huginn, Parcae and GRT. Each arm should be run at x2, x4 and x8.

Cheap check before any arm (no extra arm needed): log the RMS norm of the residual stream after each pass of
the pair, and the cosine between successive pass increments, for the existing x2/x4/x8 checkpoints. Parcae,
FLT, LT2 and Buehlmaier all show that norm growth is the failure pattern, and that the final LayerNorm hides it
from the loss curve. If the x8 seeds with the worst loss also show the largest state norm, cause 3 is confirmed
for our proxy and arms 2-4 become the priority.

1. **Gated input re-injection (scalar mix).** Before each pass p ≥ 2 of the pair, set
   h ← w·e + (1−w)·h with w = σ(a), one learned scalar (Marchenko et al. 2026 used init w = 0.99 for a converted
   pretrained model; a from-scratch init is not given in the paper). Additive variant: h ← h + e (0 params;
   Huginn says addition equals concatenation at small scale; Nguyen & Lin, Yang et al. use it).
   - Evidence: GRT ablation, re-injection −0.022 nats (small) and −0.198 (medium, short run); Qwen conversion:
     without injection later steps end 0.01-0.02 nats worse than step 1, with the scalar 0.02 nats better;
     FLT: injected looped model stays stable from 3 to 9 loops where plain looping degrades; SCSE: the
     injection/adapter package moves 95.6M PPL 117.1 → 98.6; algorithmic recall evidence (Bansal, Anil, Yang,
     Fan, Labovich) shows no-injection models degrade with more passes.
   - Extra params: 1 (or 0).
   - Risks: additive injection keeps Ā = I (marginally stable in Parcae's analysis), so with x8 the state norm
     can still grow; the convex scalar mix bounds it. Evidence at our scale is absent; the closest is the 1.2M ILR
     model, which used additive injection without an ablation.

2. **Loop-aware residual scaling on the looped pair.** Multiply the attention and MLP branch outputs of layers
   2-3 by ε = 1/r (h ← h + (1/r)·f(LN(h))), layers 1 and 4 unchanged. A learned-scalar variant (init 1/r, 2
   params per layer) is a small extension.
   - Evidence: Wang et al. 2026, 1/N beats 1/√N by 0.025 nats at N=8 and makes the best LR independent of N;
     DeepLoop (scaling rule counted by visits) improves val loss at every loop count ≥ 3 (−0.016 to −0.028
     nats); SCSE's residual step scale s = 0.35-0.5 is part of its strong controls; Tying-the-Loop: LR scaling for
     tied weights beats none (Muon). Our lr 1e-2 was tuned without loops, which fits the visit-count argument for
     the x8 seed spread.
   - Extra params: 0.
   - Risks: at x2 the shrink may slow learning of the pair; the papers loop the whole stack, not a middle pair;
     all LM results are single-seed.

3. **Parcae-style stable injection.** For each pass: h ← Ā·h + B̄·LN(e) + R(·), with Ā = exp(Δ·Diag(−exp(log A)))
   (per-channel carry in (0, 1)), B̄ = Δ·B, learned Δ ∈ R^16, B ∈ R^{16×16}, LayerNorm on e, and the learned
   readout C ∈ R^{16×16} before layer 4. OrthoInj (projecting the carried state off the injected input) is a
   refinement with +0.3-1.4% relative PPL.
   - Evidence: Parcae, pre-norm looped baselines diverge at LR ≥ 4e-4 while Parcae trains at all five LRs; val
     PPL −6.2% vs Huginn-style models, −4.3 to −9.2% vs parameter-matched Transformers (140M-1.3B); OrthoInj Table
     3 confirms Parcae-Decay as the strongest prior injection at 100M-1.6B.
   - Extra params: 576 (2.5%); 320 without C.
   - Risks: the exact input to the core R is written loosely (check the public code before implementing);
     Buehlmaier: the carry constraint is necessary, not sufficient, and a minority of inits never converge.

4. **Peri-LN on the looped pair.** x ← x + LN_out(F(LN_in(x))) for attention and MLP of layers 2-3 (2 new
   LayerNorms per layer).
   - Evidence: Closing the Loop, T=4 shared 24-layer block: Ouro-norm (Peri-LN) 1.98 vs pre-norm 2.03 loss;
     Peri-LN paper: lower loss (3.43 → 3.34 at 400M) and clearly smaller seed std over 5 seeds; Ouro trains with
     this layout; LT2 and FLT report residual growth across loops that output normalization addresses.
   - Extra params: 128 (0.55%).
   - Risks: Huginn: "at small scales ... most normalization strategies work almost equally well"; the Peri-LN
     gain in Closing the Loop is from one run.

5. **Pass-specific zero-init residual gains (timestep modulation, reduced).** Per pass p and per looped layer,
   learned vectors γ1^p, γ2^p ∈ R^16 multiply the attention and MLP branches (init 1 to keep the plain arm at
   start, or init 0 as AdaLN-zero in Xu & Sato), optionally with matching LayerNorm gains α^p.
   - Evidence: Xu & Sato, WikiText-103 looped r=24 test PPL 20.6 → 19.6 with timestep modulation (input also
     injected every loop); Hyperloop: loop-specific connection parameters 14.85 → 14.40 at 135M where LoRA rank
     32 gave only 14.77; SCSE: a plain loop-step embedding added to the input gave no gain (22M) or a loss
     (95.6M), so the modulation form, not an added embedding, is the candidate.
   - Extra params: 2×16×2 per pass = 64/128/256 for x2/x4/x8 (gains only).
   - Risks: Huginn reports a step-indexed core did not extrapolate (not tested here, fixed r); parameter count
     grows with r, so the x8 arm gets more parameters than x2.

6. **Zero-init output projections of the looped pair.** Initialize attn out_proj and mlp c_proj of layers 2-3
   to zero (block starts as identity).
   - Evidence: Loop, Think & Generalize: default init gave inconsistent behaviour across 5 seeds, zero-init gave
     consistent trends (synthetic task only).
   - Extra params: 0.
   - Risks: no LM evidence; may slow early learning of the pair.

7. **GRT gated update (reduced).** h^(p) = g ⊙ h^(p−1) + (1−g) ⊙ o with g = σ(W_g[LN(h^(p−1)); LN(e)] + b),
   b init +4, one Linear(32→16) shared across passes, plus re-injection.
   - Evidence: GRT gate −0.048 nats (small), −0.115 (medium short run), GRT seed std 0.004 vs dense 0.056.
   - Extra params: 528 + 64 (LayerNorms) ≈ 592 (2.6%); the paper's 2-layer gate form is ≈ 1,376 (6%).
   - Risks: our reduction to one linear layer is untested; gate init +4 makes each pass write about 2% at start,
     which may interact with lr 1e-2 differently from the paper's 6e-4.

Methods read and not recommended for this proxy (reason):
- Per-pass LoRA (Relaxed Recursive Transformers): gains depend on pretrained init; from-scratch LoRA gives
  ≤ 0.08 PPL at 135M (Hyperloop Table 7); rank-1 costs 576 params per pass.
- Hyperloop with n=2 streams: strong evidence, but about 1,750 params for x8 (7.5%); worth a documented
  larger-budget arm only.
- ACT / PonderNet / SUT halting, Ouro exit gate, MoR routers: change the fixed-compute protocol; ACT hurt LM
  perplexity in MoEUT's SUT comparison (21.51 → 40.50 at 44M); MoR trails vanilla at 135M.
- Saunshi regularizer: applies to models with more unique parameters.
- Random/curriculum loop counts, progressive loss, deep supervision, truncated backprop: change the training
  protocol (fixed r); they mainly serve test-time extrapolation (cause 7).
- Jacobian regularization, DEQ solvers, HRM/TRM: evidence only for equilibrium solvers or puzzle tasks.
- Muon: better than AdamW for recurrent models (McLeish), but the proxy is AdamW only by the user's choice.
- Sequence (AABB) order / intra-layer recurrence: conflicting evidence (Takase, ILR vs MoR, MoEUT); 0 params, low
  priority.
- Attention output gate (LT2): 1.3B evidence only (9.87 → 9.39 PPL); about 68 params per layer; a possible later
  arm.

Combination note: arms 1 + 2 together are the SCSE "tuned adapter" form (h ← h + s·B(h + α·W·h*)), and arm 3
already contains injection plus a stable carry. Single-factor arms first keep the attribution clean.

Stated limits: no paper reports results near 23k parameters or on TinyStories; all LM evidence is at 1.2M
(ILR) or ≥ 22M parameters. Most 2026 LM results are single-seed. Several tables (Ouro 7-13, SUT 4, MoEUT
figures, LT2 Table 2) were garbled or only in figures and are quoted only where readable.

## 11. Cause 8 — earlier-pass information is lost in one state (attention / routing over passes and depth)

Pattern: a later pass can read only the last pass's output, so information of earlier passes must survive every
rewrite of one hidden state ("information overload", MeSH abstract; Pre-LN dilution, AttnRes Sec. 2; residual RMS
about 20x after 4 loops, LT2). Fix family: let a pass or layer read the outputs of all earlier passes/layers, with
per-token weights. Two research agents read the papers below in full unless marked; raw notes with equations and
quotes: research_looped_attention.md and research_depth_attention.md (working notes of 2026-10-17, not in this
repository).

Looped / weight-shared models:
- MeSH (Yu et al., ICLR 2026, arXiv:2510.07739, code github.com/LivingFutureLab/MeSH, licence not verified).
  Per token a buffer of B = passes + 3 slots, m_0 = embedding, others 0. Per pass t its own routers:
  w_write = softmax(Linear_write^t(h)), w_read = softmax(Linear_read^t(h)); m_b += f_core(h) * w_write,b;
  h_next = sum_b m_b * w_read,b (Sec. 3.2, Eqs. 2-5). Table 1 (Pile PPL, from scratch, Pythia): 160M vanilla 11.31,
  plain loop 11.79, MeSH 11.37; 410M (4+8R2+4) 9.07 / 9.31 / 9.09; 1.4B 7.44 / 7.63 / 7.39. Table 5: static and
  dynamic learned combinations of states 2.2163 / 2.2176 vs MeSH 2.2077. Only 2-3 loops. Cost at d=16, 8 passes,
  B=11: about 3.4k params, about +8% FLOPs.
- CoTFormer (Mohtashami, Pagliardini, Jaggi, ICLR 2025, arXiv:2310.10845): each pass attends to the earlier passes'
  states of past tokens; 0 params, +90-125% FLOPs here. Table 1 (3 seeds, d=768): 12L x5 block loop 27.15 vs 26.64;
  24L x5 24.95 vs 24.48; unique layers stay better at equal depth. Think-at-Hard (arXiv:2511.08577) ablation agrees.
- Depth-Attention (arXiv:2606.05014): values mixed over earlier passes with the layer's own q, k; 0 params.
  500M, 3 loops: 2.208 -> 2.194; mixing keys alone hurts.
- InfiLoop (Li et al., arXiv:2610.11570): next state = decayed softmax over all pass outputs, 17 params; puzzles only.
- Dreamer (arXiv:2601.21582), RecurTrace (arXiv:2609.03379, fine-tune only): per-token attention over the same
  token's earlier pass outputs; Dreamer 1B-2B PPL 6.64 -> 6.41 at depth 16.
- Not useful here: FLT attention injection reads only pass t-1 (small scale no better than input injection);
  pass-1 KV sharing (PLT arXiv:2510.24824, MoR arXiv:2507.10524) costs accuracy.

Standard (non-looped) depth aggregation:
- Attention Residuals (Kimi, arXiv:2603.15031): h_l = sum_i softmax(w_l^T RMSNorm(v_i)) v_i over earlier sublayer
  outputs. 436M: 1.766 -> 1.737; fixed weights 1.749; DenseFormer no gain (1.767). ~600 params at d=16.
- DCA (arXiv:2502.06785): separate dynamic depth mixes for the Q, K, V inputs. Width 64, 12 layers, LM1B (seq 128):
  PPL 45.75 -> 42.94; the gain grows as width shrinks. About 9.1k params at d=16. No code found.
- MUDDFormer (arXiv:2502.12170): multiway dynamic dense connections; 405M PPL 11.68 -> 10.83; ~52k params at d=16.
- MoDA (arXiv:2603.15619): depth keys/values in the same softmax; pre-LN gain only 0.004 at 48 layers.
- DenseFormer (arXiv:2402.02622), ResFormer (arXiv:2410.17897), ANCRe (arXiv:2602.09009): static, cheap, mixed
  evidence across papers.
- Identity or uniform init of the depth weights was critical (MUDD, AttnRes, LIMe arXiv:2502.09245).

Stated limits: no paper trains near width 16 (smallest: 64 in DCA); most gains over a plain loop are 0.01-0.04
nats, while the proxy's loop-vs-unique gap is 0.525 nats; MeSH is the only looped method whose gain closes most of
the loop-vs-vanilla gap (88% at 160M).
