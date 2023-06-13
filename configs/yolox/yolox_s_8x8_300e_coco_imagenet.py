_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

img_scale = (640, 640)
img_scale_imagenet = (320, 320)

# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.5,
        norm_cfg=dict(type='SyncBN', momentum=0.03, eps=0.001)),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        norm_cfg=dict(type='SyncBN', momentum=0.03, eps=0.001)),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=80,
        in_channels=128,
        feat_channels=128,
        norm_cfg=dict(type='SyncBN', momentum=0.03, eps=0.001)),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
coco_dataset_type = 'CocoDataset'
imagenet_dataset_type = 'ImagenetDataset'

train_COCO_pipeline = [
    # no mosaic and mixup
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2)),
        # border=(-img_scale[0] // 2, -img_scale[1] // 2)),
        # border is only used in mosaic datasets, otherwise this raises an exception
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_Imagenet_pipeline = [
    # no mosaic and mixup
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2)),
        # border=(-img_scale[0] // 2, -img_scale[1] // 2)),
        # border is only used in mosaic datasets, otherwise this raises an exception
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale_imagenet, keep_ratio=True),
    dict(
        type='Pad',
        size=img_scale,
        # pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    #dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_COCO_dataset = dict(
    type='RepeatDataset',
    times=5,
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=coco_dataset_type,
            ann_file='data/coco/annotations/instances_train2014_seen_65_15.json',
            img_prefix='data/coco/train2014/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False, # intented because obj loss is backpropagated
        ),
        pipeline=train_COCO_pipeline))

imagenet_classes = 'data/imagenet_annotations/imagenet2012_labels_text.txt'
train_imagenet_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=imagenet_dataset_type,
        classes=imagenet_classes,
        ann_file='data/imagenet_annotations/imagenet2012_annotations_seen_65_15.json',
        img_prefix='data/imagenet/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
            # with_bbox=True is intended because we want bbox = None
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_Imagenet_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    persistent_workers=True,
    train=[train_COCO_dataset,train_imagenet_dataset],
    val=dict(
        type=coco_dataset_type,
        ann_file='data/coco/annotations/instances_val2014_seen_65_15_new.json',
        img_prefix='data/coco/val2014/',
        pipeline=test_pipeline),
    test=dict(
        type=coco_dataset_type,
        ann_file='data/coco/annotations/instances_val2014_seen_65_15_new.json',
        img_prefix='data/coco/val2014/',
        pipeline=test_pipeline))

# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(
    type="GradientCumulativeOptimizerHook",
    cumulative_iters=4,
    grad_clip=None)

find_unused_parameters = True
max_epochs = 200
num_last_epochs = 0

# we need to use load_from because of other learning rate policy than used for pretraining
load_from = "TODO" # path to COCO pretraining checkpoint
resume_from = None
interval = 1

# learning policy
lr_config = dict(
    _delete_=True,
    policy='step',
    by_epoch=True,
    step=[])
    #warmup='exp',
    #warmup_by_epoch=True,
    #warmup_iters=?,
    #warmup_ratio=0.1

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    # YOLOXModeSwitchHook turns off Mosaic and Mixup
    #dict(
    #    type='YOLOXModeSwitchHook',
    #    num_last_epochs=num_last_epochs,
    #    priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        # experiments showed that it is better to resume_from
        resume_from=load_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    metric='bbox')
log_config = dict(
    interval=50,
    hooks=[ 
        dict(type='TextLoggerHook'), 
        # note the change in mmcv/mmcv/runner/log_buffer.py:
        # we are using an averaging function that excludes all zero values from the average
        # otherwise, the bbox and obj losses are distorted when training with imagenet
        dict(type='TensorboardLoggerHook')
    ])
