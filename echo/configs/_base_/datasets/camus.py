# dataset settings
dataset_type = 'CAMUSVideoDataset'
data_root = 'data/camus'

# pipeline
pipeline = [
    dict(type='LoadNpyFile', frame_length=2, label_idxs=[0,1]),
    dict(type='PackSegMultiInputs')
]

# dataloader
train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='videos/train', seg_map_path='annotations/train'),
        pipeline=pipeline))
val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='videos/val', seg_map_path='annotations/val'),
        pipeline=pipeline))
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
        pipeline=pipeline))
val_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'], prefix='val')
test_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'], prefix='test')
