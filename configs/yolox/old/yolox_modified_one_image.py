# dont't change this line
# because the pretrained weights come from a tiny model
_base_ = './yolox_tiny_8x8_300e_coco.py'

# load pretrained backbone
model = dict(
    bbox_head=dict(
        type='YOLOXHeadModified',
        loss_cls=dict(
            type='CrossEntropyLossModified',
            use_sigmoid=False, # makes it use softmax
            reduction='sum',
            loss_weight=1.0)),
    train_cfg=dict(assigner=dict(type='SimOTAAssignerModified')),
    backbone=dict(
        init_cfg = dict(
            type='Pretrained',
            checkpoint='../../checkpoints/yolox_tiny_8x8_300e_coco_20210806_234250-4ff3b67e.pth',
            prefix='backbone.')))

optimizer=dict(
    lr = 0.01 * (1/8) * (1/8)) # we use 1 GPU instead of 8 & batch_size = 1

classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

data_root = 'data/coco/'
data = dict(
    train=dict(
        dataset=dict(
            ann_file = 'one_image.json',
            img_prefix = data_root + 'train2014/')),
    val=dict(
        ann_file = 'one_image.json',
        img_prefix = data_root + 'train2014/'),
    test=dict(
        ann_file = 'one_image.json',
        img_prefix = data_root + 'train2014/'))

log_config=dict(interval=1)
work_dir = '../../logs/yolox_modified_one_image/'