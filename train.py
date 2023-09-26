#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import wandb
from loguru import logger
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

import owdfa.algorithm as algorithm
from owdfa.datasets import create_dataloader
from owdfa.utils import get_parameters, init_wandb_workspace, setup

import better_exceptions
better_exceptions.hook()

args = get_parameters()
args = init_wandb_workspace(args)
if args.local_rank == 0:
    logger.add(f'{args.exam_dir}/train.log', level="INFO")
    logger.info(OmegaConf.to_yaml(args))


def main():
    # Distributed traning
    if args.distributed:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method="env://")
        torch.cuda.set_device(args.local_rank)
        args.world_size = dist.get_world_size()

    # Init setup
    setup(args)

    # Create dataloader
    train_dataloader = create_dataloader(args, split='train')
    test_dataloader = create_dataloader(args, split='test')

    # Resume from checkpoint
    checkpoint_dir = os.path.join(args.exam_dir, 'ckpts') if args.exam_dir else None

    model = algorithm.__dict__[args.method.name](args)

    trainer = pl.Trainer(
        strategy='ddp_find_unused_parameters_true',
        min_epochs=1,
        max_epochs=args.train.epochs,
        default_root_dir=args.exam_dir,
        callbacks=[ModelCheckpoint(
            dirpath=checkpoint_dir,
            verbose=True,
            monitor='all_nmi',
            mode='max',
            save_on_train_epoch_end=True,
        )],
        num_sanity_val_steps=1,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloader, test_dataloader)

    if args.local_rank == 0:
        wandb.finish()


if __name__ == '__main__':
    main()
