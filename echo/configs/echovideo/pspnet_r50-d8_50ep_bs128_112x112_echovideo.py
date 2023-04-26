_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/echovideo.py', 
    '../_base_/schedules/schedule_50ep.py',
    '../_base_/models/pspnet_r50-d8.py',
]

train_dataloader = dict(
    batch_size=32)

data_preprocessor = dict(
    type='SegDataPreProcessorV2',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(112, 112))

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        type='PSPHead',
        num_classes=2,
        out_channels=2,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        num_classes=2,
        out_channels=2,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
)

# boost
optim_wrapper=dict(type='AmpOptimWrapper')
cfg=dict(compile=True)

default_hooks = dict(visualization=dict(type='SegVisualizationHook'))

