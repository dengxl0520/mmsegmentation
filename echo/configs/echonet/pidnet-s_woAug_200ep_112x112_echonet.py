_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/echonet.py', 
    '../_base_/schedules/schedule_200ep.py',
    '../_base_/models/pidnet-s.py'
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size = 128,
    dataset=dict(
        pipeline=train_pipeline
    )
)

# boost
# optim_wrapper=dict(type='AmpOptimWrapper')
cfg=dict(compile=True)
