_base_ = './yolox_x_8x8_300e_coco.py'

# specify test set
ann_file_for_testing = 'data/coco/annotations/instances_val2014_unseen_65_15.json'
#ann_file_for_testing = 'data/coco/annotations/instances_val2014_seen_65_15_new.json'
#ann_file_for_testing = 'data/coco/annotations/instances_val2014_seen_unseen_65_15_new.json'

# when readding mosaic & mixup:
# - remove use_l1
# - add mosaic and mixup in train_pipeline
# - add border for RandomAffine in train_pipeline
# - add YOLOXModeSwitchHook in custom_hooks
# - set num_last_epochs = 15

# model settings
model = dict(
    bbox_head=dict(
        type='YOLOXHeadModified',
        loss_cls=dict(
            type='CrossEntropyLossModified',
            use_sigmoid=False, # makes it use softmax
            reduction='sum',
            loss_weight=1.0),
        use_zsd = True,
        use_l1=True,
        used_split="65-15", # "48-17"
        cls_out_channels = 512), # for the big CLIP model use 768, note that all ImageNet embeddings are tuned
    train_cfg=dict(assigner=dict(type='SimOTAAssignerModified')))

optimizer = dict(lr=0.01 *(1/8)*(12/8))

# dataset settings
data = dict(
    val=dict(
        ann_file=ann_file_for_testing),
    test=dict(
        ann_file=ann_file_for_testing))

work_dir = '../../logs/x_65_15_512_baseline'