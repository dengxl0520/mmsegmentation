_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/echocycle.py', 
    '../_base_/schedules/schedule_200ep.py',
    '../_base_/models/pidnet-s.py',
]

model = dict(
    type='VideoEncoderDecoder',
    input_type='video',
    data_preprocessor=dict(type='SegVideoPreProcessor'),
    neck=dict()
)
pipeline = [
    dict(type='LoadVideoAndAnnoFromFile', frame_length=10),
    dict(type='VideoGenerateEdge', edge_width=4),
    dict(type='PackSegVideoInputs')
]

train_dataloader = dict(
    batch_size = 128,
    dataset=dict(
        pipeline=pipeline
    )
)

# boost
cfg=dict(compile=True)
