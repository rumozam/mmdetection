import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger


#### Set Parameters ### 
num_gpus = 3
load_from_checkpoint = False
######################


def configure_logging(cfg, config, timestamp):
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(config)))
    # init the logger before other steps
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: True')
    logger.info(f'Config:\n{cfg.pretty_text}')

    return logger, meta


def main(num_gpus):
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    model = "yolox_modified.py"
    config = "./configs/yolox/" + model

    cfg = Config.fromfile(config)

    #Working dir is model name without the ".py"
    cfg.work_dir = '../../logs/' + model[:-3] + "/" + timestamp

    if load_from_checkpoint:
        checkpoint = "/data/vilab14/logs/yolox_modified/full_dataset_30epochs/latest.pth"

        # remove ema_ from state dict
        # -> see .dev_scripts/gather_models.py process_checkpoint()
        modified_checkpoint = checkpoint[:-4] + '_modified.pth'
        checkpoint = torch.load(checkpoint, map_location='cpu')
        for key in list(checkpoint['state_dict']):
                if key.startswith('ema_'):
                    checkpoint['state_dict'].pop(key)
        torch.save(checkpoint, modified_checkpoint, _use_new_zipfile_serialization=False)

        cfg.resume_from = modified_checkpoint

    # Init distributed code execution
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    local_rank = args.local_rank
    print(f"\n\n\n\nLocal rank: {local_rank}\n\n\n\n")
    
    init_dist('pytorch', local_rank, **cfg.dist_params)
    _, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)

    # The original learning rate (LR) is set for 8-GPU training.
    # Adopt learning rate according to the number of used GPUs
    cfg.optimizer.lr = cfg.optimizer.lr * (num_gpus/3)

    # have a look at the final config used for training
    #print(cfg.pretty_text)
    
    logger, meta = configure_logging(cfg,config,timestamp)

    # set random seeds
    seed = init_random_seed(None)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: False')
    set_random_seed(seed, deterministic=False)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if not load_from_checkpoint:
        model.init_weights()

    datasets = [build_dataset(cfg.data.train)]

    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(model, datasets, cfg, distributed=True, validate=True, timestamp=timestamp, meta=meta)

if __name__ == '__main__':
    main(num_gpus)