#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import argparse
import lightning.pytorch as pl
from glob import glob
from omegaconf import OmegaConf
from loguru import logger

import owdfa.algorithm as algorithm
from owdfa.datasets import create_dataloader

import warnings
warnings.filterwarnings("ignore")


def main():
    # set configs
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='./configs/eval.yaml')
    parser.add_argument('--exam_id', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--output_log', type=str, default='eval.log')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--distributed', type=int, default=0)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    local_config = OmegaConf.load(args.config)
    for k, v in local_config.items():
        setattr(args, k, v)

    os.environ['TORCH_HOME'] = args.torch_home

    # search the checkpoint file according EXAM ID
    if args.exam_id:
        ckpt_path = glob(f'wandb/*{args.exam_id}/ckpts/*.ckpt')
        if len(ckpt_path) >= 1:
            ckpt_path = sorted(ckpt_path)
            args.ckpt_path = ckpt_path[-1]
    exam_dir = os.path.dirname(os.path.dirname(args.ckpt_path))

    # add log file
    if len(args.output_log) > 0:
        logger.add(f'{exam_dir}/{args.output_log}', level="INFO")

    # load dataset
    test_dataloader = create_dataloader(args, split=args.split)

    method = algorithm.__dict__[args.method.name](args)

    trainer = pl.Trainer(default_root_dir=exam_dir)
    trainer.validate(method, test_dataloader, args.ckpt_path)


if __name__ == '__main__':
    main()
