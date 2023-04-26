_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_100ep.py',
    '../_base_/models/pspnet_r50-d8.py'
]

# dataset settings
dataset_type = 'EchonetVideoDataset'
data_root = 'data/echonet/echocycle'

# pipeline
pipeline = [
    dict(type='LoadNpyFile', frame_length=2, label_idxs=['0','1']),
    dict(type='VideoGenerateEdge', edge_width=4),
    dict(type='PackSegMultiInputs')
]

# dataloader
train_dataloader = dict(
    batch_size=8,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='videos/train', seg_map_path='annotations/train'),
        pipeline=pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='videos/val', seg_map_path='annotations/val'),
        pipeline=pipeline))
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
        pipeline=pipeline))
val_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator

data_preprocessor = dict(
    type='SegDataPreProcessorV2',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(112,112))

model = dict(
    type='VideoEncoderDecoder',
    input_type='video',
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        type='PSPHead',
        num_classes=2,
        out_channels=2,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
)

# boost
optim_wrapper=dict(type='AmpOptimWrapper')
cfg=dict(compile=True)
