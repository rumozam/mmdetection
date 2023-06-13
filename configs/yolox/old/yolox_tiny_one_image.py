_base_ = './yolox_tiny_8x8_300e_coco.py'

# load pretrained backbone
model = dict(
    backbone=dict(
        init_cfg = dict(
            type='Pretrained',
            checkpoint='/data/vilab16/checkpoints/yolox_tiny_8x8_300e_coco_20210806_234250-4ff3b67e.pth',
            prefix='backbone.')))

optimizer=dict(
    lr = 0.01 / 8) # we use 1 GPU instead of 8

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
            img_prefix = data_root + 'train2014/',
            classes = classes)),
    val=dict(
        ann_file = 'one_image.json',
        img_prefix = data_root + 'train2014/',
        classes = classes),
    test=dict(
        ann_file = 'one_image.json',
        img_prefix = data_root + 'train2014/',
        classes = classes))


log_config=dict(interval=1)
work_dir = '../../logs/yolox_tiny_one_image/'