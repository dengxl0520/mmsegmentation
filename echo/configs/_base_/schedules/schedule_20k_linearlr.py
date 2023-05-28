# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=20000)
]
# training schedule for 20k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', log_metric_by_epoch=False, interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, max_keep_ckpts=5),
    sampler_seed=dict(type='DistSamplerSeedHook'))
