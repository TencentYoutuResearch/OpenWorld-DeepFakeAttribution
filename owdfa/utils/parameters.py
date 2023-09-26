#!/usr/bin/env python
# coding: utf-8
import os
import argparse
import omegaconf
from loguru import logger


def flatten_dict(dictionary, exclude=None, delimiter='.'):
    if exclude is None:
        exclude = []
    flat_dict = dict()
    for key, value in dictionary.items():
        if isinstance(value, dict) and key not in exclude:
            flatten_value_dict = flatten_dict(value, exclude, delimiter)
            for k, v in flatten_value_dict.items():
                flat_dict[f"{key}{delimiter}{k}"] = v
        else:
            flat_dict[key] = value
    return flat_dict


def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def get_parameters(to_dict=False, config=None, **new_kwargs):
    # priority: cmd args > new_kwargs > dict in config
    # `--config` in cmd args can overwrite `config`

    # (1) the base parser, introduce `config`
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config', type=str,
                        default=config or 'cfg.yaml')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--sync-bn', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    # parse the above cmd options
    args_tmp = parser.parse_known_args()[0]
    args_tmp.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args_tmp.distributed = int(os.environ['WORLD_SIZE']) > 1
    if args_tmp.debug:
        args_tmp.train.epochs = 2
    args_tmp_dict = vars(args_tmp)

    try:
        oc_cfg = omegaconf.OmegaConf.load(args_tmp.config)
    except Exception as e:
        logger.warning(f'config path [{args_tmp.config}] is not valid.')
        raise e
    oc_cfg.merge_with(args_tmp_dict)

    # (2) append items from new_kwargs, new_kwargs has higher priority
    if new_kwargs:
        for k in new_kwargs:
            if k in oc_cfg:
                logger.warning(
                    f'{k} from `new_kwargs` found in original conf, will keep the one in `new_kwargs`')
        oc_cfg.merge_with(omegaconf.OmegaConf.create(new_kwargs))

    oc_cfg_dict = omegaconf.OmegaConf.to_container(oc_cfg, resolve=True)
    # turn chained dict to flatten dict, turn {a: {b: {c: 1}}} to a.b.c = 1
    oc_cfg_dict_flatten = flatten_dict(oc_cfg_dict)

    # (3) add all the k-v items from `oc_cfg_dict_flatten` to argparse
    # dest change dotted names to be connected with '___', as dotted dest name is not supported by argparse
    for k, v in oc_cfg_dict_flatten.items():
        if k in args_tmp_dict:
            continue
        # not using the default store_true syntax
        # use lambda func so that we can use `python a.py --switch True`
        if isinstance(v, bool):
            parser.add_argument('--{}'.format(k), dest=k.replace('.', '___'),
                                type=lambda x: (str(x).lower() == 'true'), default=v)
        elif isinstance(v, list) or isinstance(v, tuple):
            parser.add_argument(
                '--{}'.format(k), dest=k.replace('.', '___'), type=type(v[0]), default=v, nargs='+')
        else:
            # None type default changed to str
            parser.add_argument('--{}'.format(k), dest=k.replace('.', '___'),
                                type=str if v is None else type(v), default=v)
    parser.add_argument('-h', '--help', action='help',
                        help=('show this help message and exit'))
    args = parser.parse_args()

    var_args = vars(args)
    for k, v in var_args.items():
        ori_k = k.replace('___', '.')
        sub_ks = ori_k.split('.')
        # turn back to nested dict
        # turn a.b.c = 1 to {a: {b: {c: 1}}}
        nested_set(oc_cfg_dict, sub_ks, v)

    oc_cfg.merge_with(oc_cfg_dict)

    # print(OmegaConf.to_yaml(oc_cfg))
    if to_dict:
        return omegaconf.OmegaConf.to_container(oc_cfg, resolve=True)
    return oc_cfg


if __name__ == '__main__':
    args = get_parameters()
    print(args)
