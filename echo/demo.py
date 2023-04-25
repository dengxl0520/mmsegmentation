from mmengine.registry import init_default_scope
from mmseg.datasets import CityscapesDataset

init_default_scope('mmseg')

data_root = 'data/cityscapes/'
data_prefix=dict(img_path='leftImg8bit/train', seg_map_path='gtFine/train')
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

dataset = CityscapesDataset(data_root=data_root, data_prefix=data_prefix, test_mode=False, pipeline=train_pipeline)


for i in range(len(dataset)):
    print(dataset[i])
    print(dataset[i]['data_samples'].gt_sem_seg)
