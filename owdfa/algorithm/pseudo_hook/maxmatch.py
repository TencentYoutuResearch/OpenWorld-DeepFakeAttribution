import torch

from .base import PseudoHook


class MaxHook(PseudoHook):
    """
    Default selection in MaxMatch
    """

    def __init__(self, **kwargs):
        self.eta = kwargs['eta'] if 'eta' in kwargs else 1.0

    @torch.no_grad()
    def masking(self, logits_x_ulb, **kwargs):
        probs_x_ulb = logits_x_ulb.detach()
        max_probs, _ = torch.max(probs_x_ulb, dim=-1)
        return torch.ones_like(max_probs)
