"""2 Top-level forward pass.

    tokens -> Embedding -> B1 -> B2(routed bank) -> B_mid -> B3(routed bank) -> B4
           -> Norm -> LM head

B1, B_mid and B4 are shared and run exactly once.  B2 and B3 each hold ONE expert
body plus a bank of hypernetworks and a router that selects between them (5, 11).
Counters reset at the B2 -> B3 boundary because those are different parameter
sets (11.6, 4.1).

There is one model.  An earlier two-stage plan (pretrain a single expert alone,
then assemble a routed bank) has been dropped entirely -- what follows is the
whole architecture, not a stage of it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attnres import AttnRes
from .blocks import SharedBlock
from .config import Config
from .hypernet import Hypernet
from .kda import KDA


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
        self.b1, self.bmid, self.b4 = SharedBlock(c), SharedBlock(c), SharedBlock(c)
        self.e2, self.e3 = RoutedBlock(c), RoutedBlock(c)
        self.final = nn.RMSNorm(c.d_model, eps=c.eps)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)

        self.apply(_init)
        for m in self.modules():          # restore inits the blanket pass clobbered
            if isinstance(m, KDA):
                m.reset_gate()
            elif isinstance(m, Hypernet):
                m.reset_out()
            elif isinstance(m, AttnRes):
                nn.init.zeros_(m.w)
        if c.tie_embeddings:
            self.head.weight = self.embed.weight

    def forward(self, ids, labels=None):
        from .router import AuxLoss

        aux = AuxLoss(self.c.n_hypernets, ids.device) if self.training else None
        h = self.b1(self.embed(ids))
        h, d2 = self.e2(h, aux)
        h = self.bmid(h)
        h, d3 = self.e3(h, aux)            # 11.6 counters reset: different parameters
        logits = self.head(self.final(self.b4(h)))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(),
                                   labels[:, 1:].flatten(), ignore_index=-100)
            if aux is not None:
                a, self.aux_stats = aux.value()
                self.aux_stats["lm"] = float(loss)
                loss = loss + self.c.lambda_aux * a
        return logits, loss, (d2, d3)

    # ---- parameter accounting -------------------------------------------- #

    def shared_parameters(self):
        """9 EWC-protected: everything not owned by a routed block (e2/e3)."""
        routed = {id(p) for m in (self.e2, self.e3) for p in m.parameters()}
        return [(n, p) for n, p in self.named_parameters() if id(p) not in routed]

    def param_report(self):
        n = lambda mods: sum(p.numel() for m in mods for p in m.parameters())
        hyp = n([self.e2.expert.hypers, self.e3.expert.hypers])
        body = n([self.e2.expert, self.e3.expert]) - hyp
        total = sum(p.numel() for p in dict(self.named_parameters()).values())
        rows = [("shared scaffold (B1+B_mid+B4)", n([self.b1, self.bmid, self.b4])),
                ("expert body (B2+B3)", body),
                (f"{self.c.n_hypernets} hypernets (B2+B3)", hyp),
                ("routers", n([self.e2.router, self.e3.router])),
                ("embedding + LM head", self.embed.weight.numel()
                 * (1 if self.c.tie_embeddings else 2))]
        return rows, total, total
