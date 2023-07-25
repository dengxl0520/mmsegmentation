# dataset settings
dataset_type = 'EchonetVideoDataset'
data_root = 'data/echonet/echocycle'
# pipeline
train_pipeline = [
    dict(type='LoadNpyFile', frame_length=10, label_idxs=[0,9]),
    dict(type='VideoPhotoMetricDistortion'),
    dict(type='VideoRandomFlip', prob=0.5),
    dict(type='PackSegMultiInputs')
]
test_pipeline = [
    dict(type='LoadNpyFile', frame_length=10, label_idxs=[0,9]),
    dict(type='PackSegMultiInputs')
]
# dataloader
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='videos/train', seg_map_path='annotations/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='videos/val', seg_map_path='annotations/val'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='videos/test', seg_map_path='annotations/test'),
        pipeline=test_pipeline))
val_evaluator = dict(
    type='EchoMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'], prefix='val')
test_evaluator = dict(
    type='EchoMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'], prefix='test')
