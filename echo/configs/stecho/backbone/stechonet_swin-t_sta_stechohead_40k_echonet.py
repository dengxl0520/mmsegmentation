_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/stechonet_r50_sta_stechohead.py',
    '../../_base_/datasets/echonet.py',
    '../../_base_/schedules/schedule_40k_cosinelr_sgd_1e-2.py'
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
depths = [2, 2, 6, 2]
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
        neck=dict(
            _delete_=True,
            type='STAttention',
            input_shape=dict(
                res2=dict(channels=96, stride=4),
                res3=dict(channels=192, stride=8),
                res4=dict(channels=384, stride=16),
                res5=dict(channels=768, stride=32)),
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=1024,
            transformer_enc_layers=6,
            conv_dim=256,
            mask_dim=256,
            norm='GN',
            transformer_in_features=['res3', 'res4', 'res5'],
            common_stride=4,
            temporal_attn_ksize_offset=1,
            attention_type='STAttention',
        ),
)

