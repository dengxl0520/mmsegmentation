

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'  # noqa
class_weight = [1,1]

model = dict(
    type='VideoEncoderDecoder',
    data_preprocessor=dict(type='SegVideoPreProcessor'),
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='PIDNet',
        in_channels=3,
        channels=32,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    decode_head=dict(
        type='PIDHead',
        in_channels=128,
        channels=128,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
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
                loss_weight=1.0)
        ]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataloader
train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='EchonetVideoDataset',
        data_root='data/echonet/echocycle',
        data_prefix=dict(img_path='videos/train', seg_map_path='annotations/train'),
        pipeline=[
            dict(
                type='LoadVideoAndAnnoFromFile',
                frame_length=2),
            dict(type='GenerateEdge', edge_width=4),
            dict(type='PackSegVideoInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='EchonetVideoDataset',
        data_root='data/echonet/echocycle',
        data_prefix=dict(img_path='videos/val', seg_map_path='annotations/val'),
        pipeline=[
            dict(
                type='LoadVideoAndAnnoFromFile',
                frame_length=2),
            dict(type='PackSegVideoInputs')
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='EchonetVideoDataset',
        data_root='data/echonet/echocycle',
        data_prefix=dict(img_path='videos/test', seg_map_path='annotations/test'),
        pipeline=[
            dict(
                type='LoadVideoAndAnnoFromFile',
                frame_length=2),
            dict(type='PackSegVideoInputs')
        ]))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mDice','mFscore'])
test_evaluator = val_evaluator

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None)
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
default_scope = 'mmseg'
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer')
log_level = 'INFO'
load_from = None
resume = False
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=2),
    sampler_seed=dict(type='DistSamplerSeedHook'))

# cfg  Based epoch
param_scheduler = dict(type='MultiStepLR', milestones=[6, 8])
train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=50)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

randomness = dict(seed=1234)
