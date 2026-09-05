"""2 Top-level forward pass.

    tokens -> Embedding -> L1 -> [L2 + L3](recurrent, routed) -> L4
           -> Norm -> LM head

Four KDA layers, four FFNs. L1 and L4 run once; L2+L3 are one
recurrent block holding a bank of hypernetworks and a router that selects between
them (5, 11).  Only L3's FFN has generated weights.

There is one model.  An earlier two-stage plan (pretrain a single expert alone,
then assemble a routed bank) has been dropped entirely -- what follows is the
whole architecture, not a stage of it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .blocks import Layer
from .config import Config
from .hypernet import AdapterBasis


def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)


class CinnamonModel(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        from .routed import RoutedBlock

        self.c = c
        self.embed = nn.Embedding(c.vocab, c.d_model)
        self.l1 = Layer(c)
        self.expert = RoutedBlock(c)          # L2 + L3
        self.l4 = Layer(c)
        self.final = nn.RMSNorm(c.d_model, eps=c.eps)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)

        self.apply(_init)
        for m in self.modules():          # restore inits the blanket pass clobbered
            if isinstance(m, AdapterBasis):
                m.reset_out()
        if c.tie_embeddings:
            self.head.weight = self.embed.weight

    def forward(self, ids, labels=None, router_warmup=False):
        from .router import AuxLoss

        aux = AuxLoss(self.c.n_hypernets, ids.device) if self.training else None
        h = self.l1(self.embed(ids))
        h, depth = self.expert(h, aux, router_warmup=router_warmup)
        h = self.final(self.l4(h))

        loss, logits = None, None
        if labels is not None and self.c.loss_chunk:
            loss = _chunked_ce(self.head, h[:, :-1].flatten(0, 1),
                               labels[:, 1:].flatten(), self.c.loss_chunk)
        else:
            logits = self.head(h)
        if labels is not None:
            if loss is None:
                loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(),
                                       labels[:, 1:].flatten(), ignore_index=-100)
            if aux is not None:
                a, self.aux_stats = aux.value()
                self.aux_stats["lm"] = float(loss)
                loss = loss + self.c.lambda_aux * a
                if router_warmup:
                    self.aux_stats["div"] = float(self.expert._div_loss)
                    loss = loss + self.c.lambda_diversity * self.expert._div_loss
        return logits, loss, depth

    def forward_activations(self, h0, *, recurrence_steps=16):
        """Activation-distillation mapping without an embedding or LM head.

        ``h0`` is Qwen's residual stream before its first transformer block.
        C1 is Cinnamon's existing first layer; C2 is one uninterrupted pass
        through the existing routed recurrent block, returning each internal
        state; C3 is Cinnamon's existing final layer.  The final RMSNorm and
        tied language-model head are intentionally outside this mapping because
        the target ``h36`` is Qwen's residual output after block 36.
        """
        if h0.ndim != 3 or h0.shape[-1] != self.c.d_model:
            raise ValueError(
                f"h0 must have shape [batch, seq, {self.c.d_model}], got {tuple(h0.shape)}")
        if recurrence_steps != 16:
            raise ValueError("activation distillation requires exactly 16 C2 passes")
        if self.c.c_max2 < recurrence_steps:
            raise ValueError(
                f"c_max2={self.c.c_max2} cannot provide {recurrence_steps} supervised passes")

        # Pilot artifacts are BF16, while callers may keep Cinnamon parameters in
        # FP32 for validation or in a mixed-precision training configuration.
        h0 = h0.to(dtype=self.l1.n1.weight.dtype)
        c1 = self.l1(h0)
        _, _, c2_states = self.expert(
            c1, aux=None, return_states=True, steps=recurrence_steps)
        c3 = self.l4(c2_states[-1])
        return c1, c2_states, c3

    # ---- parameter accounting -------------------------------------------- #

    def param_report(self):
        n = lambda mods: sum(p.numel() for m in mods for p in m.parameters())
        e = self.expert.expert
        hyp, basis = n([e.hypers]), n([e.basis])
        body = n([e]) - hyp - basis
        total = sum(p.numel() for p in dict(self.named_parameters()).values())
        emb = self.embed.weight.numel() * (1 if self.c.tie_embeddings else 2)
        rows = [("L1 + L4", n([self.l1, self.l4])),
                ("L2 + L3 body", body),
                (f"{self.c.n_hypernets} hypernet trunks", hyp),
                ("shared adapter basis", basis),
                ("router", n([self.expert.router])),
                ("embedding + LM head", emb)]
        # Third value is NON-EMBEDDING, which is the budgeted number.  It used to
        # return `total` again; both callers discarded it, so the bug was invisible.
        return rows, total, total - emb


def _chunked_ce(head, h, tgt, chunk):
    """Cross-entropy without materialising [N, vocab].

    Each tile is checkpointed, so its logits are freed after the forward and
    recomputed in backward -- peak is one tile, not the whole sequence.  Nothing
    on the attention side helps here: compression shrinks what a token attends
    TO, while this is what every token PREDICTS, and that stays vocab-wide.
    """
    n = (tgt != -100).sum().clamp(min=1)
    tile = lambda hc, tc: F.cross_entropy(
        head(hc).float(), tc, ignore_index=-100, reduction="sum")
    total = 0.0
    for i in range(0, h.shape[0], chunk):
        total = total + checkpoint(tile, h[i:i + chunk], tgt[i:i + chunk],
                                   use_reentrant=False)
    return total / n
