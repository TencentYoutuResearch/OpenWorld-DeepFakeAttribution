import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from loguru import logger

from owdfa.algorithm.ssl import SSLModel
from owdfa.losses import entropy, MarginLoss
from owdfa.utils import AverageMeter, update_meter


class MPSL(SSLModel):
    def __init__(self, args):
        super(MPSL, self).__init__(args)
        self.ce = MarginLoss(m=-1 * args.method.MPSL.m, s=1)
        self.mpv = nn.BCELoss()
        self.eta1 = args.method.MPSL.eta1
        self.eta2 = args.method.MPSL.eta2
        self.eta3 = args.method.MPSL.eta3
        self.batch_size = args.train.batch_size

    def get_loss_names(self):
        loss_name = ['total_loss', 'ce_loss',
                     'mpv_loss', 'fcp_loss', 'en_loss']
        return loss_name

    def on_train_epoch_start(self):
        SSLModel.on_train_epoch_start(self)
        self.seq = {
            0: 'global',
            1: 'freq1',
            2: 'freq2',
            3: 'freq3',
            4: 'un_freq',
            5: 'part1',
            6: 'part2',
            7: 'part3',
            8: 'un_part',
            9: 'either',
            10: 'both',
        }
        meter = [self.seq[i] + '_known' for i in range(11)] + \
            [self.seq[i] + '_novel' for i in range(11)]
        self.label_meter = {feat: AverageMeter(feat, ':.2f')
                            for feat in meter}
        self.label_value = {feat: 0 for feat in meter}
        self.label_size = {feat: 0 for feat in meter}

    def on_train_epoch_end(self):
        SSLModel.on_train_epoch_start(self)
        results = {key: value.avg for key, value in self.label_meter.items()}
        for k, v in results.items():
            logger.info(f'{k}: {v * 100:.2f}')

    def training_step(self, batch, batch_idx):
        loss = SSLModel.training_step(self, batch, batch_idx)
        for key, value in self.label_value.items():
            if self.label_size[key] == 0:
                continue
            update_meter(
                self.label_meter[key],
                value/self.label_size[key],
                self.label_size[key],
                self.args.distributed,
            )
            self.label_value[key] = self.label_size[key] = 0
        return loss

    def _update_label_meter(self, i, target, target_ulb, y, pos_idx):
        la = (target_ulb == y[pos_idx]).float()
        la_mask = (target_ulb <= target.max()).float()
        self.label_value[self.seq[i] + '_known'] += (la * la_mask).sum()
        self.label_value[self.seq[i] + '_novel'] += (la * (1 - la_mask)).sum()
        self.label_size[self.seq[i] + '_known'] += la_mask.sum()
        self.label_size[self.seq[i] + '_novel'] += (1 - la_mask).sum()

    def _get_pair(self, target, f_g, f_p, labeled_len, total_len, target_ulb, y):
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
        f_freq, f_p_all = f_p
        # 3. frequency feature
        freq_sim_all = []
        for f_p in f_freq:
            freq_detach = f_p.detach()
            freq_norm = freq_detach / torch.norm(freq_detach, 2, 1, keepdim=True)
            # [B, C, H/2 * W/2] -> [H/2 * W/2, B, C] -> [H/2 * W/2, B, B] -> [B, B, H/2 * W/2]
            freq_cosine_dist = torch.bmm(freq_norm.permute(
                2, 0, 1), freq_norm.permute(2, 1, 0)).permute(1, 2, 0)
            # [B, B, H/2 * W/2] -> [B, B, 1]
            freq_sim = freq_cosine_dist.sum(dim=2)
            freq_sim_all.append(freq_sim)
        # 4. part feature
        part_sim_all = []
        for f_p in f_p_all:
            part_feat = f_p.detach()
            part_norm = torch.norm(part_feat, 2, 1)
            feat_p_norm = part_feat / torch.norm(part_feat, 2, 1, keepdim=True)
            # [B, C, H * W] -> [H * W, B, C] -> [H * W, B, B] -> [B, B, H * W]
            #              |-> [H * W, C, B] 
            part_cosine_dist = torch.bmm(feat_p_norm.permute(
                2, 0, 1), feat_p_norm.permute(2, 1, 0)).permute(1, 2, 0)
            part_norm = part_norm / torch.norm(part_norm, 2, 1, keepdim=True)
            part_sim = (part_cosine_dist * part_norm.repeat(total_len,
                        1, 1).permute(1, 0, 2)).sum(dim=2)
            part_sim_all.append(part_sim)

        # Global similarity
        i = 0
        _, pos_idx = torch.topk(global_sim[labeled_len:, :], 2, dim=1)
        self._update_label_meter(
            i, target, target_ulb, y, pos_idx[:, 1].cpu().numpy().flatten().tolist())
        i += 1

        # Frequency filter
        mask_freq = torch.zeros_like(pos_idx[:, 0])
        for freq_sim in freq_sim_all:
            vals, pos_idx_t = torch.topk(freq_sim[labeled_len:, :], 2, dim=1)
            choose_k = 2  # this parameter should be fine-tuned with different task
            max_pos = torch.topk(freq_sim[:, pos_idx[:, 1]], choose_k, dim=0)[
                0][choose_k-1]
            mask_freq = (vals[:, 1] - max_pos).ge(0).float() + mask_freq
            self._update_label_meter(
                i, target, target_ulb, y, pos_idx_t[:, 1].cpu().numpy().flatten().tolist())
            i += 1
            
        mask_1 = (mask_freq.ge(1)).float()
        mask_0 = 1 - mask_1
        pos_idx_1 = (pos_idx[:, 1] * mask_1).cpu().numpy()
        pos_idx_0 = (pos_idx[:, 0] * mask_0).cpu().numpy()
        pos_idx_l = (pos_idx_1 + pos_idx_0).flatten().tolist()
        self._update_label_meter(i, target, target_ulb, y, pos_idx_l)
        i += 1

        # Part filter
        mask_part = torch.zeros_like(pos_idx[:, 0])
        for part_sim in part_sim_all:
            vals, pos_idx_t = torch.topk(part_sim[labeled_len:, :], 2, dim=1)
            choose_k = 2  # this parameter should be fine-tuned with different task
            max_pos = torch.topk(part_sim[:, pos_idx[:, 1]], choose_k, dim=0)[
                0][choose_k-1]
            mask_part = (vals[:, 1] - max_pos).ge(0).float() + mask_part

            self._update_label_meter(
                i, target, target_ulb, y, pos_idx_t[:, 1].cpu().numpy().flatten().tolist())
            i += 1

        mask_1 = (mask_part.ge(1)).float()
        mask_0 = 1 - mask_1
        pos_idx_1 = (pos_idx[:, 1] * mask_1).cpu().numpy()
        pos_idx_0 = (pos_idx[:, 0] * mask_0).cpu().numpy()
        pos_idx_l = (pos_idx_1 + pos_idx_0).flatten().tolist()
        self._update_label_meter(i, target, target_ulb, y, pos_idx_l)
        i += 1

        mask_1 = (mask_freq.ge(1) + mask_part.ge(1)).float()
        mask_0 = 1 - mask_1
        mask_u = mask_1
        pos_idx_1 = (pos_idx[:, 1] * mask_1).cpu().numpy()
        pos_idx_0 = (pos_idx[:, 0] * mask_0).cpu().numpy()
        pos_idx_l = (pos_idx_1 + pos_idx_0).flatten().tolist()
        self._update_label_meter(i, target, target_ulb, y, pos_idx_l)
        i += 1

        mask_1 = (mask_freq.ge(1) * mask_part.ge(1)).float()
        mask_0 = 1 - mask_1
        mask_x = mask_1
        pos_idx_1 = (pos_idx[:, 1] * mask_1).cpu().numpy()
        pos_idx_0 = (pos_idx[:, 0] * mask_0).cpu().numpy()
        pos_idx = (pos_idx_1 + pos_idx_0).flatten().tolist()
        pos_pairs.extend(pos_idx)
        self._update_label_meter(i, target, target_ulb, y, pos_idx)
        return pos_pairs, (mask_u, mask_x)

    def ce_loss(self, output, target, labeled_len):
        return self.ce(output[:labeled_len], target)

    def mpv_loss(self, prob, **kwargs):
        target = kwargs['target']
        feat = kwargs['feat']
        feat_p = kwargs['feat_p']
        labeled_len = kwargs['labeled_len']
        total_len = kwargs['total_len']
        target_ulb = kwargs['target_ulb']
        y = kwargs['y']

        pos_pairs, mask_1 = self._get_pair(
            target, feat, feat_p, labeled_len, total_len, target_ulb, y)
        tar_prob = prob[pos_pairs, :]
        if len(tar_prob) < self.batch_size:
            return torch.zeros(1).cuda(), mask_1
        tar_sim = torch.bmm(prob.view(self.batch_size, 1, -1),
                            tar_prob.view(self.batch_size, -1, 1)).squeeze()
        tar_ones = torch.ones_like(tar_sim)
        return self.mpv(tar_sim, tar_ones), mask_1

    def fcp_loss(self, output, prob_ul, labeled_len, masks):
        mask_u, mask_x = masks
        rand_prob = F.gumbel_softmax(
            output[labeled_len:].detach(), tau=1, hard=False)
        target_u_pl = torch.argmax(rand_prob, dim=1)
        max_prob_pl = prob_ul.gather(1, target_u_pl.view(-1, 1)).squeeze()
        # Filter out the too-high confidence pseudo-labels
        # hard_mask = mask_u
        # max_prob_pl = hard_mask * max_prob_pl
        fcp_loss = (F.cross_entropy(
            output[labeled_len:], rand_prob, reduction='none') * max_prob_pl).mean()
        return fcp_loss

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
        mpv_loss, mask = self.mpv_loss(prob, **kwargs)
        loss_map['mpv_loss'] = mpv_loss
        # Confident-based Soft Pseudo label
        prob_ul = F.softmax(output[labeled_len:], dim=1)
        fcp_loss = self.fcp_loss(output, prob_ul, labeled_len, mask)
        loss_map['fcp_loss'] = fcp_loss
        # Entropy loss
        en_loss = self.en_loss(prob)
        loss_map['en_loss'] = en_loss
        # Total loss
        total_loss = ce_loss + self.eta1 * mpv_loss + \
            self.eta2 * fcp_loss - self.eta3 * en_loss
        loss_map['total_loss'] = total_loss
        return loss_map
