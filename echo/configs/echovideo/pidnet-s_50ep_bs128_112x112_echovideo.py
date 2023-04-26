_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/echovideo.py', 
    '../_base_/schedules/schedule_50ep.py',
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
    data_preprocessor=data_preprocessor,
)
train_dataloader = dict(
    batch_size = 128,
)

# boost
# optim_wrapper=dict(type='AmpOptimWrapper')
cfg=dict(compile=True)
# vis
default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
)