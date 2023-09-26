import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from owdfa.algorithm.ssl import SSLModel
from owdfa.losses import entropy, MarginLoss


class CPL(SSLModel):
    def __init__(self, args):
        super(CPL, self).__init__(args)
        self.ce = MarginLoss(m=-1 * args.method.CPL.m, s=1)
        self.glv = nn.BCELoss()
        self.eta1 = args.method.CPL.eta1
        self.eta2 = args.method.CPL.eta2
        self.eta3 = args.method.CPL.eta3
        self.batch_size = args.train.batch_size

    def get_loss_names(self):
        loss_name = ['total_loss', 'ce_loss',
                     'glv_loss', 'csp_loss', 'en_loss']
        return loss_name

    def _get_pair(self, target, f_g, f_p, labeled_len, total_len):
        pos_pairs = []
        target_np = target.cpu().numpy()
        # label part
        for ind in range(labeled_len):
            target_i = target_np[ind]
            idxs = np.where(target_np == target_i)[0]
            if len(idxs) == 1:
                pos_pairs.append(idxs[0])
            else:
                selec_idx = np.random.choice(idxs, 1)
                while selec_idx == ind:
                    selec_idx = np.random.choice(idxs, 1)
                pos_pairs.append(int(selec_idx))
        # unlabel part
        # 1. global feature
        feat_detach = f_g.detach()
        feat_g_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
        global_cosine_dist = torch.mm(feat_g_norm, feat_g_norm.t())
        global_sim = global_cosine_dist
        # 2. local feature
        part_feat = f_p.detach()
        part_norm = torch.norm(part_feat, 2, 1)
        feat_p_norm = part_feat / torch.norm(part_feat, 2, 1, keepdim=True)
        part_cosine_dist = torch.bmm(feat_p_norm.permute(
            2, 0, 1), feat_p_norm.permute(2, 1, 0)).permute(1, 2, 0)
        # 3. partial mask
        part_norm = part_norm / torch.norm(part_norm, 2, 1, keepdim=True)
        part_sim = (part_cosine_dist * part_norm.repeat(total_len,
                    1, 1).permute(1, 0, 2)).sum(dim=2)

        # Global similarity & Part filter
        _, pos_idx = torch.topk(global_sim[labeled_len:, :], 2, dim=1)
        vals, _ = torch.topk(part_sim[labeled_len:, :], 2, dim=1)
        choose_k = 2  # this parameter should be fine-tuned with different task
        max_pos = torch.topk(part_sim[:, pos_idx[:, 1]], choose_k, dim=0)[
            0][choose_k-1]
        mask_1 = (vals[:, 1] - max_pos).ge(0).float()
        mask_0 = (vals[:, 1] - max_pos).lt(0).float()
        pos_idx_1 = (pos_idx[:, 1] * mask_1).cpu().numpy()
        pos_idx_0 = (pos_idx[:, 0] * mask_0).cpu().numpy()
        pos_idx = (pos_idx_1 + pos_idx_0).flatten().tolist()
        pos_pairs.extend(pos_idx)
        return pos_pairs, mask_1

    def ce_loss(self, output, target, labeled_len):
        return self.ce(output[:labeled_len], target)

    def glv_loss(self, prob, **kwargs):
        target = kwargs['target']
        feat = kwargs['feat']
        feat_p = kwargs['feat_p']
        labeled_len = kwargs['labeled_len']
        total_len = kwargs['total_len']

        pos_pairs, mask_1 = self._get_pair(
            target, feat, feat_p, labeled_len, total_len)
        tar_prob = prob[pos_pairs, :]
        if len(tar_prob) < self.batch_size:
            return torch.zeros(1).cuda(), mask_1
        tar_sim = torch.bmm(prob.view(self.batch_size, 1, -1),
                            tar_prob.view(self.batch_size, -1, 1)).squeeze()
        tar_ones = torch.ones_like(tar_sim)
        return self.glv(tar_sim, tar_ones), mask_1

    def csp_loss(self, output, prob_ul, labeled_len, mask_1):
        rand_prob = F.gumbel_softmax(
            output[labeled_len:].detach(), tau=1, hard=False)
        target_u_pl = torch.argmax(rand_prob, dim=1)
        max_prob_pl = prob_ul.gather(1, target_u_pl.view(-1, 1)).squeeze()
        csp_loss = (F.cross_entropy(
            output[labeled_len:], rand_prob, reduction='none') * max_prob_pl * mask_1).mean()
        return csp_loss

    def en_loss(self, prob):
        return entropy(torch.mean(prob, 0))

    def loss(self, **kwargs):
        labeled_len = kwargs['labeled_len']
        output = kwargs['output']
        target = kwargs['target']

        loss_map = {}
        # Cross Entropy loss
        ce_loss = self.ce_loss(output, target, labeled_len)
        loss_map['ce_loss'] = ce_loss
        # Global Local Voting loss
        prob = F.softmax(output, dim=1)
        glv_loss, mask = self.glv_loss(prob, **kwargs)
        loss_map['glv_loss'] = glv_loss
        # Confident-based Soft Pseudo label
        prob_ul = F.softmax(output[labeled_len:], dim=1)
        csp_loss = self.csp_loss(output, prob_ul, labeled_len, mask)
        loss_map['csp_loss'] = csp_loss
        # Entropy loss
        en_loss = self.en_loss(prob)
        loss_map['en_loss'] = en_loss
        # Total loss
        total_loss = ce_loss + self.eta1 * glv_loss + \
            self.eta2 * csp_loss - self.eta3 * en_loss
        loss_map['total_loss'] = total_loss
        return loss_map
