_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/stechonet_r50_sta_stechoheadv3.py',
    '../../_base_/datasets/echonet.py',
    '../../_base_/schedules/schedule_20k_cosinelr_sgd_1e-2.py'
]
size=(128,128)
data_preprocessor = dict(
    type='SegVideoPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=size)
model = dict(data_preprocessor=data_preprocessor)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'))
