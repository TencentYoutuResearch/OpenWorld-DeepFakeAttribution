from collections import Counter
from copy import deepcopy
import torch

from .base import PseudoHook


class FlexMatchHook(PseudoHook):
    """
    Adaptive Thresholding in FlexMatch
    """

    def __init__(self, **kwargs):
        self.num_classes = kwargs['num_classes']
        self.ulb_dest_len = kwargs['ulb_dest_len']
        self.p_cutoff = kwargs['p_cutoff'] if 'p_cutoff' in kwargs else 0.95
        self.thresh_warmup = kwargs['thresh_warmup'] if 'thresh_warmup' in kwargs else True
        self.eta = kwargs['eta'] if 'eta' in kwargs else 1.0

        self.selected_label = torch.ones(
            (self.ulb_dest_len,), dtype=torch.long, ) * -1
        self.classwise_acc = torch.zeros((self.num_classes,))

    @torch.no_grad()
    def update(self, **kwargs):
        pseudo_counter = Counter(self.selected_label.tolist())

        if max(pseudo_counter.values()) < self.ulb_dest_len:  # not all(5w) -1
            if self.thresh_warmup:
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / \
                        max(pseudo_counter.values())
            else:
                wo_negative_one = deepcopy(pseudo_counter)
                if -1 in wo_negative_one.keys():
                    wo_negative_one.pop(-1)
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / \
                        max(wo_negative_one.values())

    @torch.no_grad()
    def masking(self, logits_x_ulb, idx_ulb, **kwargs):
        if not self.selected_label.is_cuda:
            self.selected_label = self.selected_label.to(logits_x_ulb.device)
        if not self.classwise_acc.is_cuda:
            self.classwise_acc = self.classwise_acc.to(logits_x_ulb.device)

        probs_x_ulb = logits_x_ulb.detach()
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1)
        mask = max_probs.ge(
            self.p_cutoff * (self.classwise_acc[max_idx] / (2. - self.classwise_acc[max_idx])))  # convex
        select = max_probs.ge(self.p_cutoff)
        mask = mask.to(max_probs.dtype)

        # update
        if idx_ulb[select == 1].nelement() != 0:
            self.selected_label[idx_ulb[select == 1]] = max_idx[select == 1]
        self.update()
        return mask
