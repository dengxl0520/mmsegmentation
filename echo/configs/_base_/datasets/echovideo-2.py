_base_ = './echovideo.py'

# pipeline
pipeline = [
    dict(type='LoadNpyFile', frame_length=2, label_idxs=[0,1]),
    dict(type='VideoGenerateEdge', edge_width=4),
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
