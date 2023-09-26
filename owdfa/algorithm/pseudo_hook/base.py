import torch
import torch.nn.functional as F


class PseudoHook():
    """PseudoHook

    Hook for pseudo label loss

    Parameters
    ----------
    eta: float
        Weight of pseudo label loss
    """

    def __init__(self, **kwargs):
        raise NotImplementedError

    def masking(self, logits_x_ulb, **kwargs):
        raise NotImplementedError

    def pce_loss(self, **kwargs):
        """Compute pseudo label loss

        Parameters
        ----------
        output: torch.Tensor
            Output of model
        labeled_len: int
            Length of labeled data
        idx_ulb: torch.Tensor
            Index of unlabeled data

        Returns
        -------
        pce_loss: torch.Tensor
            Pseudo label loss
        """
        output = kwargs['output']
        labeled_len = kwargs['labeled_len']
        prob_ul = F.softmax(output[labeled_len:], dim=1)
        mask_pl = self.masking(prob_ul, **kwargs)
        _, targets_u_pl = torch.max(prob_ul.detach(), dim=1)
        pce_loss = (F.cross_entropy(
            output[labeled_len:], targets_u_pl, reduction='none') * mask_pl).mean()
        return pce_loss * self.eta
