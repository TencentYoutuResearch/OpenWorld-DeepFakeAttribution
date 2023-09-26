from loguru import logger
import torch.utils.data as data

from owdfa.datasets.dfa import DFA
from owdfa.datasets.utils import create_data_transforms, MultilabelBalancedRandomSampler


def create_dataloader(args, split, loader='torch'):
    if args.dataset.loader == 'torch':
        return create_torch_dataloader(args, split)
    else:
        logger.error(f'Unknown loader: {args.dataset.loader}')


def create_torch_dataloader(args, split):
    num_workers = args.num_workers if 'num_workers' in args else 8
    balance_sample = args.balance_sample if 'balance_sample' in args else False
    batch_size = getattr(args, split).batch_size

    transform = create_data_transforms(args.transform, split)
    kwargs = getattr(args.dataset, args.dataset.name)
    dataset = eval(args.dataset.name)(
        split=split, transform=transform, **kwargs)

    sampler = None

    if args.distributed:
        sampler = data.DistributedSampler(dataset)

    if balance_sample and split == 'train':
        sampler = MultilabelBalancedRandomSampler(dataset)

    if args.debug:
        dataset = data.Subset(dataset, range(0, 100))
        sampler = None
        logger.warning(
            'Distributed dataset sampler is disabled in debug mode!')

    shuffle = True if sampler is None and split == 'train' else False

    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 sampler=sampler,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 )

    return dataloader
