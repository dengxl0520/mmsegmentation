_base_ = './pidnet-s_multigpm.py'
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-l_imagenet1k_20230306-67889109.pth'  # noqa
model = dict(
    backbone=dict(
        channels=64,
        ppm_channels=112,
        num_stem_blocks=3,
        num_branch_blocks=4,
        init_cfg=dict(checkpoint=checkpoint_file)),
    neck=dict(
        type='MultiGPM',
        d_model=128,
    ),
    decode_head=dict(in_channels=256, channels=256))
