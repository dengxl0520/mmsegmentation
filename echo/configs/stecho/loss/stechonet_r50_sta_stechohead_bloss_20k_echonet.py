_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/stechonet_r50_sta_stechohead.py',
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
model = dict(
    data_preprocessor=data_preprocessor,
    neck=dict(transformer_enc_layers=6),
    decode_head=dict(
        type='STEchoHeadwithBloss',
        loss_decode=[
            dict(
                type='CrossEntropyLoss', 
                loss_name='loss_ce', 
                use_sigmoid=True,
                reduction='mean',
                loss_weight=1.0),
            dict(
                type='BinaryDiceLoss', 
                loss_name='loss_dice', 
                eps=1.0,
                loss_weight=1.0),
            dict(
                type='BoundaryLossV2' ,
                loss_name='loss_boundary',
                loss_weight=0.01
            )
        ]
    )
)