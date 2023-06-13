from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, init_random_seed, set_random_seed
import time

#### Parameters ### 
model = "yolox_modified_coco_imagenet_10percent.py"
######################


config = "./configs/yolox/" + model
cfg = Config.fromfile(config)

# Modify config
cfg.gpu_ids = range(1)
cfg.data.samples_per_gpu = 1
cfg.data.workers_per_gpu = 1
cfg.work_dir = '../../logs/debugging'

# set random seeds
seed = init_random_seed(0)
set_random_seed(seed, deterministic=True)
cfg.seed = seed

# have a look at the final config used for training
#print(cfg.pretty_text)

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.init_weights()

# Train the detector
train_detector(model, datasets, cfg, validate=True)