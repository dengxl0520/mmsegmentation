_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/echovideo-2.py', 
    '../_base_/schedules/schedule_20k_cosinelr.py',
    '../_base_/models/pidnet-s.py'
]

data_preprocessor = dict(
    type='SegDataPreProcessorV2',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(112, 112))

model = dict(
    type='VideoEncoderDecoder',
    input_type='video',
    supervised='sup',
    data_preprocessor=data_preprocessor,
)

# pipeline
pipeline = [
    dict(type='LoadNpyFile', frame_length=2, label_idxs=[0,1]),
    dict(type='VideoGenerateEdge', edge_width=1),
    dict(type='PackSegMultiInputs')
]

# dataloader
train_dataloader = dict(
    dataset=dict(pipeline=pipeline)
)
val_dataloader = dict(
    dataset=dict(pipeline=pipeline)
)
test_dataloader = dict(
    dataset=dict(pipeline=pipeline)
)
