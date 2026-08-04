"""9 Elastic weight consolidation over the shared scaffold.

Phase 1 trains one domain at a time, so the shared blocks are the only thing every
expert competes over.  EWC anchors them to where the previous domain left them,
weighted by how much that domain cared:

    L_i = L_LM,i + (lambda/2) * sum_j F_j (theta_j - theta_j*)^2

Expert-owned parameters are deliberately unprotected -- an expert should be free
to specialise.  Domain 1 has no anchor, so this is inert for the first expert; it
is written now so expert 2 does not arrive to a missing mechanism.

Known and accepted (9): EWC prevents forgetting but not drift.  E_1 is tuned
against the scaffold as it was at domain 1 and never updated again, so later
experts are better matched to the final scaffold.  Mitigated by domain ordering.
"""
import torch


class EWC:
    """Diagonal Fisher.  14.4 decided: diagonal only, lambda 5000, 200 batches."""

    def __init__(self, lam: float = 5000.0, fisher_batches: int = 200):
        self.lam = lam
        self.fisher_batches = fisher_batches
        self.anchor = {}     # name -> theta*
        self.fisher = {}     # name -> F

    def penalty(self, model) -> torch.Tensor:
        if not self.anchor:
            return torch.zeros((), device=next(model.parameters()).device)
        total = 0.0
        for n, p in model.shared_parameters():
            if n in self.anchor:
                total = total + (self.fisher[n] * (p - self.anchor[n]).pow(2)).sum()
        return 0.5 * self.lam * total

    @torch.enable_grad()
    def consolidate(self, model, batches, log=print):
        """Estimate the diagonal Fisher as E[(d log p / d theta)^2] and re-anchor.

        Uses the model's own loss gradient, which is the empirical Fisher -- the
        standard cheap approximation; the exact Fisher would need sampling from the
        model's predictive distribution instead of the observed labels.
        """
        shared = dict(model.shared_parameters())
        fisher = {n: torch.zeros_like(p) for n, p in shared.items()}
        model.eval()
        n_done = 0
        for i, (x, y) in enumerate(batches):
            if i >= self.fisher_batches:
                break
            model.zero_grad(set_to_none=True)
            _, loss, _ = model(x, labels=y)
            loss.backward()
            for n, p in shared.items():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)
            n_done += 1
        model.zero_grad(set_to_none=True)
        model.train()
        if not n_done:
            log("EWC: no batches, keeping previous anchor")
            return
        self.fisher = {n: f / n_done for n, f in fisher.items()}
        self.anchor = {n: p.detach().clone() for n, p in shared.items()}
        log(f"EWC: consolidated {len(self.anchor)} shared tensors over {n_done} batches")

    def state_dict(self):
        return {"lam": self.lam, "fisher_batches": self.fisher_batches,
                "anchor": self.anchor, "fisher": self.fisher}

    def load_state_dict(self, s):
        self.lam, self.fisher_batches = s["lam"], s["fisher_batches"]
        self.anchor, self.fisher = s["anchor"], s["fisher"]
