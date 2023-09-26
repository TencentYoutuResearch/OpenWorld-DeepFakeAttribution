import torch
import torch.nn.functional as F
import wandb

from owdfa.algorithm.sl import SLModel
import owdfa.algorithm.pseudo_hook as pseudo_hook
from owdfa.utils import AverageMeter, update_meter


class SSLModel(SLModel):
    def __init__(self, args):
        super().__init__(args)
        hook = None
        if 'pseudo' in args and args.pseudo.hook != 'None':
            hook = pseudo_hook.__dict__[args.pseudo.hook](ulb_dest_len=len(
                self.train_dataloader.dataset), **args.pseudo.params)
        self.hook = hook

    def get_loss_names(self) -> list:
        """Obtain the name of all losses

        Returns
        -------
        loss_names: list
            List with the name of all losses
        """
        raise NotImplementedError

    def loss(self, **kwargs) -> dict:
        """Compute all losses

        Parameters
        ----------
        labeled_len: int
            Length of labeled data
        output: torch.Tensor
            Output of model
        feat: torch.Tensor
            Feature of model
        target: torch.Tensor
            Ground truth
        device: torch.device
            Device
        kwargs: dict
            Other arguments

        Returns
        -------
        loss: dict
            Dictionary with all losses
        """
        raise NotImplementedError

    def on_train_epoch_start(self):
        self.train_losses = {loss: AverageMeter(loss, ':.2f')
                             for loss in self.get_loss_names()}

    def training_step(self, batch, batch_idx):
        tag = batch['tag']
        images = batch['image']
        targets = batch['target']
        idxs = batch['idx']

        x1 = images[tag == 1]
        x2 = images[tag == 2]
        target = targets[tag == 1]
        idx_ulb = idxs[tag == 2]
        labeled_len = len(target)
        total_len = len(targets)
        x = torch.cat([x1, x2], dim=0)

        output, (f_g, f_p) = self.encoder(x)

        # Calculate the loss using the loss function
        loss_map = self.loss(labeled_len=labeled_len, total_len=total_len, output=output,
                             feat=f_g, feat_p=f_p, target=target, idx_ulb=idx_ulb)
        # Get the total loss from the loss map
        loss = loss_map['total_loss']
        for key, value in loss_map.items():
            update_meter(
                self.train_losses[key], value, self.args.train.batch_size, self.args.distributed)
        for ls in self.train_losses.values():
            self.log(ls.name, ls.avg, on_step=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        results = {key: value.avg for key, value in self.train_losses.items()}
        wandb.log(results, step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        tags = batch['tag']
        images = batch['image']
        targets = batch['target']

        image = images[tags != 0]
        target = targets[tags != 0]
        tag = tags[tags != 0]

        output, _ = self.encoder(image)
        prob = F.softmax(output, dim=1)
        conf, pred = prob.max(1)

        self.val_step_outputs['tags'].extend(tag)
        self.val_step_outputs['preds'].extend(pred)
        self.val_step_outputs['label'].extend(target)
        self.val_step_outputs['conf'].extend(conf)
        return pred
