import torch.nn.functional as F
import lightning.pytorch as pl
import wandb
from loguru import logger
from timm.models import resume_checkpoint

from owdfa.utils import gather_tensor, val_stat
import owdfa.encoder as encoder
import owdfa.optimizers as optimizers
import owdfa.schedulers as schedulers
import owdfa.losses as losses


class SLModel(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.encoder = encoder.__dict__[args.model.name](**args.model.params)
        if args.model.resume is not None:
            resume_checkpoint(self.encoder, args.model.resume)
            if args.local_rank == 0:
                logger.info(f'resume model from {args.model.resume}')

    def configure_optimizers(self):
        optimizer = optimizers.__dict__[self.args.optimizer.name](
            self.encoder.parameters(), **self.args.optimizer.params)
        scheduler = schedulers.__dict__[self.args.scheduler.name](
            optimizer, **self.args.scheduler.params)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        criterion = losses.__dict__[
            self.args.loss.name](**self.args.loss.params)

        images = batch['image']
        targets = batch['target']

        output = self.encoder(images)
        loss = criterion(output, targets)
        self.log('ce_loss', loss, on_step=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_step_outputs = {
            'tags': [],
            'preds': [],
            'label': [],
            'conf': [],
        }

    def validation_step(self, batch, batch_idx):
        tags = batch['tag']
        images = batch['image']
        targets = batch['target']

        image = images[tags != 0]
        target = targets[tags != 0]
        tag = tags[tags != 0]

        output = self.encoder(image)
        prob = F.softmax(output, dim=1)
        conf, pred = prob.max(1)

        self.val_step_outputs['tags'].extend(tag)
        self.val_step_outputs['preds'].extend(pred)
        self.val_step_outputs['label'].extend(target)
        self.val_step_outputs['conf'].extend(conf)
        return pred

    def on_validation_epoch_end(self):
        logger.info(f'Epoch-{self.current_epoch} validation finished')
        y_tags = gather_tensor(self.val_step_outputs['tags'],
                               dist_=self.args.distributed, to_numpy=True).astype(int)
        y_pred = gather_tensor(self.val_step_outputs['preds'],
                               dist_=self.args.distributed, to_numpy=True).astype(int)
        y_label = gather_tensor(self.val_step_outputs['label'],
                                dist_=self.args.distributed, to_numpy=True).astype(int)
        y_conf = gather_tensor(self.val_step_outputs['conf'],
                               dist_=self.args.distributed, to_numpy=True)
        results = val_stat(y_tags, y_pred, y_label, y_conf)

        self.log_dict(results, on_epoch=True)
        try:
            wandb.log(results, step=self.current_epoch)
        except:
            pass

    def predict_step(self, batch, batch_idx):
        images = batch['image']
        feature, _ = self.encoder.forward_features(images)
        feature = feature.detach().cpu().numpy()

        return {
            'feature': feature,
            'tag': batch['tag'],
            'target': batch['target'],
            'img_path': batch['img_path'],
        }
