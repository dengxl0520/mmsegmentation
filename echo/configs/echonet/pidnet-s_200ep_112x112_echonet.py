_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/echonet.py', 
    '../_base_/schedules/schedule_200ep.py',
    '../_base_/models/pidnet-s.py'
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(512, 256),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(112,112), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
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
# vis
default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
)