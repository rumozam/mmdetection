_base_ = './yolox_s_8x8_300e_coco.py'

# for 1 GPU
optimizer = dict(lr=0.01 *(1/8))

data = dict(
    train=dict(
        dataset=dict(
            ann_file='data/coco/annotations/instances_train2014_seen_65_15_percent_10.json')))

work_dir = '../../logs/yolox_s_coco_10percent/'