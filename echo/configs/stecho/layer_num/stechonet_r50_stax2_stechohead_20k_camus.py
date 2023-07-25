_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/stechonet_r50_sta_stechohead.py',
    '../../_base_/datasets/camus.py',
    '../../_base_/schedules/schedule_20k_cosinelr_sgd_1e-2.py'
]
size=(320,320)
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
    neck=dict(transformer_enc_layers=2)
)