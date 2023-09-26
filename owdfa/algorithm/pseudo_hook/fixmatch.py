import torch

from .base import PseudoHook


class FixMatchHook(PseudoHook):
    """
    Threshold in FixMatch
    """

    def __init__(self, **kwargs):
        self.p_cutoff = kwargs['p_cutoff'] if 'p_cutoff' in kwargs else 0.95
        self.eta = kwargs['eta'] if 'eta' in kwargs else 1.0

    @torch.no_grad()
    def masking(self, logits_x_ulb, **kwargs):
        probs_x_ulb = logits_x_ulb.detach()
        max_probs, _ = torch.max(probs_x_ulb, dim=-1)
        mask = max_probs.ge(self.p_cutoff)
        mask = mask.to(max_probs.dtype)
        return mask
