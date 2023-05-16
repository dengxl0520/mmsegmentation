# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# learning policy
param_scheduler = [
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=True,
         begin=0,
         end=10),
    dict(type='CosineAnnealingLR',
         T_max=800,
         by_epoch=True,
         begin=10,
         end=100,
         convert_to_iter_based=True,)
]
# training schedule for 100ep
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=50)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=50),
    sampler_seed=dict(type='DistSamplerSeedHook'))
