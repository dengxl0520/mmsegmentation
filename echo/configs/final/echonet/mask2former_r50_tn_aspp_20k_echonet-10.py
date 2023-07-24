
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

# random seed
randomness = dict(seed=1234)

# compile
cfg=dict(compile=True)

# dataset settings
dataset_type = 'EchonetVideoDataset'
data_root = 'data/echonet/echocycle'
# pipeline
train_pipeline = [
    dict(type='LoadNpyFile', frame_length=10, label_idxs=[0,9]),
    dict(type='VideoPhotoMetricDistortion'),
    dict(type='VideoRandomFlip', prob=0.5),
    dict(type='PackSegMultiInputs')
]
test_pipeline = [
    dict(type='LoadNpyFile', frame_length=10, label_idxs=[0,9]),
    dict(type='PackSegMultiInputs')
]
# dataloader
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='videos/train', seg_map_path='annotations/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='videos/val', seg_map_path='annotations/val'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='videos/test', seg_map_path='annotations/test'),
        pipeline=test_pipeline))
val_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'], prefix='val')
test_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'], prefix='test')


data_preprocessor = dict(
    type='SegVideoPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(128, 128))
num_classes = 2
model = dict(
    type='SemiVideoEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNet',
        depth=50,
        deep_stem=False,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=False),
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='TemporalNeck',
        input_shape={
            'res2':{
                "channels":256,
                "stride":4
            },
            'res3':{
                "channels":512,
                "stride":8
            },
            'res4':{
                "channels":1024,
                "stride":16
            },
            'res5':{
                "channels":2048,
                "stride":32
            },
        },
    ),    
    decode_head=dict(
        type='ASPPHead',
        in_channels=256,
        channels=512,
        num_classes=num_classes,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        norm_cfg=dict(type='SyncBN', requires_grad=False),
        align_corners=False,
        loss_decode=[
            dict(type='DiceLoss', loss_weight=5.0),
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=[1, 1],
                loss_weight=5.0),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))
# learning policy
param_scheduler = [
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=2000),
    dict(type='CosineAnnealingLR',
         T_max=18000,
         by_epoch=False,
         begin=2000,
         end=20000,)
]

# training schedule for 90k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=16)
