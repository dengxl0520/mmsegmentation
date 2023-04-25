_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/echocycle.py',
    '../_base_/schedules/schedule_200ep.py',
    '../_base_/models/pidnet-s.py',
]

model = dict(
    type='VideoEncoderDecoder',
    input_type='video',
    data_preprocessor=dict(
        type='SegVideoPreProcessor',
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
    ),
)
train_pipeline = [
    dict(type='LoadVideoAndAnnoFromFile', frame_length=10),
    dict(type='VideoRandomFlip', prob=0.5),
    # dict(type='VideoPhotoMetricDistortion'),
    dict(type='VideoGenerateEdge', edge_width=4),
    dict(type='PackSegVideoInputs')
]
pipeline = [
    dict(type='LoadVideoAndAnnoFromFile', frame_length=10),
    dict(type='VideoGenerateEdge', edge_width=4),
    dict(type='PackSegVideoInputs')
]
train_dataloader = dict(batch_size=128, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=pipeline))
test_dataloader = dict(dataset=dict(pipeline=pipeline))

# boost
cfg = dict(compile=True)

# vis
# default_hooks = dict(
#     visualization=dict(type='SegNpyVisualizationHook', draw=True, interval=1))
