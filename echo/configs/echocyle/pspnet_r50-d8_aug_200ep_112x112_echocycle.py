_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/echocycle.py', 
    '../_base_/schedules/schedule_200ep.py',
    '../_base_/models/pspnet_r50-d8.py',
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

train_pipeline = [
    dict(type='LoadVideoAndAnnoFromFile', frame_length=10),    
    dict(type='RandomCrop', crop_size=(112,112), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegVideoInputs')
]
test_pipeline = [
    dict(type='LoadVideoAndAnnoFromFile', frame_length=10),
    dict(type='PackSegVideoInputs')
]
train_dataloader = dict(
    batch_size = 128,
    dataset=dict(
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline
    )
)

test_dataloder = dict(
    dataset=dict(
        pipeline=test_pipeline
    )
)

# boost
optim_wrapper=dict(type='AmpOptimWrapper')
cfg=dict(compile=True)
