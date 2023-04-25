_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/echonet.py', 
    '../_base_/schedules/schedule_200ep.py',
    '../_base_/models/pspnet_r50-d8.py',
]

model = dict(
    data_preprocessor=dict(size=(112, 112)),
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

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]


# boost
optim_wrapper=dict(type='AmpOptimWrapper')
cfg=dict(compile=True)

default_hooks = dict(visualization=dict(type='SegVisualizationHook'))

