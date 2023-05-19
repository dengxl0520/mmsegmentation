_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/echovideo-2.py', 
    '../_base_/schedules/schedule_40k_cosinelr.py',
    '../_base_/models/pidnet-s.py'
]

data_preprocessor = dict(
    type='SegDataPreProcessorV2',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(112, 112))

model = dict(
    type='VideoEncoderDecoder',
    input_type='video',
    supervised='sup',
    data_preprocessor=data_preprocessor,
)
