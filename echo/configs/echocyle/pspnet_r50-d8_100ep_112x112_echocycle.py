_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/echocycle.py', 
    '../_base_/schedules/schedule_100ep.py',
    '../_base_/models/pspnet_r50-d8.py'
]

model = dict(
    type='VideoEncoderDecoder',
    input_type='video',
    data_preprocessor=dict(type='SegVideoPreProcessor'),
    decode_head=dict(
        type='PSPHead',
        num_classes=2,
        out_channels=2,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
)

# boost
optim_wrapper=dict(type='AmpOptimWrapper')
cfg=dict(compile=True)
