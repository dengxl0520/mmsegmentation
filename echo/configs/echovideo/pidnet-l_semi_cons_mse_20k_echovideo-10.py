_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/echovideo-10.py', 
    '../_base_/schedules/schedule_20k_cosinelr.py',
    '../_base_/models/pidnet-l_semi.py'
]

data_preprocessor = dict(
    type='SegDataPreProcessorV2',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(112, 112))

class_weight = [1, 1]
model = dict(
    type='VideoEncoderDecoder',
    input_type='video',
    supervised='semisup',
    data_preprocessor=data_preprocessor,
        decode_head=dict(loss_decode=[
        dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=class_weight,
            loss_weight=0.4),
        dict(
            type='OhemCrossEntropy',
            thres=0.9,
            min_kept=131072,
            class_weight=class_weight,
            loss_weight=1.0),
        dict(type='BoundaryLoss', loss_weight=20.0),
        dict(
            type='OhemCrossEntropy',
            thres=0.9,
            min_kept=131072,
            class_weight=class_weight,
            loss_weight=1.0),
        dict(type='MSEConsistencyLoss', loss_weight=1.0),
    ]),
)

custom_hooks = [
    dict(type='LossHook', interval=1)
]
