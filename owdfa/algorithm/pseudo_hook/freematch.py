import torch

from .base import PseudoHook


class FreeMatchHook(PseudoHook):
    """
    SAT in FreeMatch
    """

    def __init__(self, **kwargs):
        self.num_classes = kwargs['num_classes']
        self.m = kwargs['momentum'] if 'momentum' in kwargs else 0.999
        self.eta = kwargs['eta'] if 'eta' in kwargs else 1.0

        self.p_model = torch.ones((self.num_classes))  # / self.num_classes
        self.label_hist = torch.ones((self.num_classes))  # / self.num_classes
        self.time_p = self.p_model.mean()

    def value(self):
        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        return self.time_p * mod

    @torch.no_grad()
    def update(self, **kwargs):
        probs_x_ulb = kwargs['probs_x_ulb']
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1, keepdim=True)
        self.time_p = self.time_p * self.m + \
            (1 - self.m) * torch.quantile(max_probs, 0.8)
        self.p_model = self.p_model * self.m + \
            (1 - self.m) * probs_x_ulb.mean(dim=0)
        hist = torch.bincount(
            max_idx.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype)
        self.label_hist = self.label_hist * self.m + \
            (1 - self.m) * (hist / hist.sum())

    @torch.no_grad()
    def masking(self, logits_x_ulb, **kwargs):
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(logits_x_ulb.device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(logits_x_ulb.device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(logits_x_ulb.device)

        probs_x_ulb = logits_x_ulb.detach()
        self.update(probs_x_ulb=probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        mask = max_probs.ge(self.time_p * mod[max_idx]).to(max_probs.dtype)
        return mask
