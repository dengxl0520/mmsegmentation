from mmseg.apis import init_model, inference_model, show_result_pyplot

config_path = 'work_dirs/pidnet-s_1xb1_20k_112x112_echonet/pidnet-s_1xb1_20k_112x112_echonet.py'
checkpoint_path = 'work_dirs/pidnet-s_1xb1_20k_112x112_echonet/iter_20000.pth'
img_path = 'data/echonet/images/val/0X1A030EFDD45062FA_70.png'

# 在 CPU 上的初始化模型并加载权重
model = init_model(config_path, checkpoint_path, 'cpu')

result = inference_model(model, img_path)

# vis_image = show_result_pyplot(model, img_path, result, out_file='work_dirs/result.png', show=False)


import mmcv
video = mmcv.VideoReader('/data/dengxiaolong/EchoNet-Dynamic/Videos/0X46DDEFA74CDB0EBD.avi')
for i in range(len(video)):
   result = inference_model(model, video[i])
   vis_image = show_result_pyplot(model, video[i], result, out_file='work_dirs/result.png', show=False)

