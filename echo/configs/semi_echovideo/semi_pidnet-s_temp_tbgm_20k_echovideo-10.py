class_weight = [1,1]
norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'  # noqa
data_preprocessor = dict(
    type='SegVideoPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(112, 112))
model = dict(
    type='SemiVideoEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='PIDNet',
        in_channels=3,
        channels=32,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(
        type='TemporalBoundaryGuidedMoudle',
        in_channels=128,
        channels=128,
        kernel_size=1,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU',inplace=True),
    ),
    decode_head=dict(
        type='PIDHeadSemi',
        in_channels=128,
        channels=128,
        num_classes=2,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=class_weight,
                loss_weight=0.4),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=20.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0),        
            dict(type='TempConsistencyLoss', loss_weight=1.0),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer, clip_grad=None)
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
# training schedule for 20k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', log_metric_by_epoch=False, interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, max_keep_ckpts=5),
    sampler_seed=dict(type='DistSamplerSeedHook'))

# dataset settings
dataset_type = 'EchonetVideoDataset'
data_root = 'data/echonet/echocycle'
# pipeline
train_pipeline = [
    dict(type='LoadNpyFile', frame_length=10, label_idxs=[0,9]),
    dict(type='VideoGenerateEdge', edge_width=2),
    dict(type='PackSegMultiInputs')
]
test_pipeline = [
    dict(type='LoadNpyFile', frame_length=10, label_idxs=[0,9]),
    dict(type='VideoGenerateEdge', edge_width=2),
    dict(type='PackSegMultiInputs')
]
# dataloader
train_dataloader = dict(
    batch_size=64,
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
    batch_size=64,
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
