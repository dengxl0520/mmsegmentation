from mmengine.registry import init_default_scope
from mmseg.datasets import EchonetDataset

init_default_scope('mmseg')

def echocycle():
    data_root = 'data/echonet/echocycle'     
    data_prefix=dict(img_path='videos/train', seg_map_path='annotations/train') 
    from mmseg.datasets import EchonetVideoDataset
    train_pipeline = [
        dict(
            type='LoadVideoAndAnnoFromFile',
            frame_length=10),
        dict(type='PackSegVideoInputs')
    ]

    dataset = EchonetVideoDataset(data_root=data_root, data_prefix=data_prefix,pipeline=train_pipeline)

    print(dataset.__getitem__(0))
    

def echonet():
    data_root = 'data/echonet/'
    data_prefix=dict(img_path='images/train', seg_map_path='annotations/train')

    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='PackSegInputs')
    ]


    dataset = EchonetDataset(data_root=data_root, pipeline=train_pipeline, data_prefix=data_prefix)

    print(dataset.__getitem__(0))

    # config = 'echo/pspnet_r50-d8_112x112_20k_echonet.py'

    # for i in range(len(dataset)):
    #     # print(dataset[i])

    #     # print(dataset[i]['inputs'])
    #     # print(dataset[i]['inputs'].sum())

    #     print(dataset[i]['data_samples'].gt_sem_seg)
    #     print(dataset[i]['data_samples'].gt_sem_seg.data)
    #     data = dataset[i]['data_samples'].gt_sem_seg.data
    #     print(dataset[i]['data_samples'].gt_sem_seg.data.sum())
    #     print(dataset[i]['data_samples'].gt_sem_seg.shape)


echocycle()
# echonet()