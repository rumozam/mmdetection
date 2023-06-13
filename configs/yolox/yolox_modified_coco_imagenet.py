_base_ = './yolox_l_8x8_300e_coco_imagenet.py'

# specify test set
ann_file_for_testing = 'data/coco/annotations/instances_val2014_unseen_65_15.json'
#ann_file_for_testing = 'data/coco/annotations/instances_val2014_seen_65_15_new.json'
#ann_file_for_testing = 'data/coco/annotations/instances_val2014_seen_unseen_65_15_new.json'

# model settings
model = dict(
    bbox_head=dict(
        type='YOLOXHeadModified',
        loss_cls=dict(
            type='CrossEntropyLossModified',
            use_sigmoid=False, # makes it use softmax
            reduction='sum',
            loss_weight=1.0),
        norm_cfg=dict(type='SyncBN', momentum=0.03, eps=0.001),
        use_zsd = True,
        use_l1=True,
        used_split="65-15", # "48-17"
        use_tuned_imagenet_labels=True,
        cls_out_channels = 512), #768
    train_cfg=dict(assigner=dict(type='SimOTAAssignerModified')))

# 0.0005 = lr COCO pretraining ends with when using lr = 0.01 for pretraining
# TODO try larger learning rate
optimizer = dict(lr=0.0005 *(2/8) *(2/8))

# dataset settings
img_scale = (640, 640)
img_scale_imagenet = (320, 320)

coco_dataset_type = 'CocoDataset'
imagenet_dataset_type = 'ImagenetDataset'

train_COCO_pipeline = [
    # no Mixup, Mosaic
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2)),
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
    # no Mixup, Mosaic
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale_imagenet, keep_ratio=True),
    dict(
        type='Pad',
        size=img_scale,
        #pad_to_square=True,
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

data = dict(
    train=[train_COCO_dataset,train_imagenet_dataset],
    val=dict(
        ann_file=ann_file_for_testing),
    test=dict(
        ann_file=ann_file_for_testing))

work_dir = '../../logs/final/yolox_coco_imagenet_syncBatchNorm_cocoRatio_cumulativeGrads/'
