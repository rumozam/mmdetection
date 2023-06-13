_base_ = './yolox_modified_coco.py'

# learning rate is set in yolox_modified_coco.py
# test set is specified in yolox_modified_coco.py

data = dict(
    train=dict(
        dataset=dict(
            ann_file='data/coco/annotations/instances_train2014_seen_65_15_percent_1.json')))

work_dir = '../../logs/yolox_modified_coco_1percent/'