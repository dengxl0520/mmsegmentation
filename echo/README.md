# Echo

## TODO List:

*一般*
- [ ] 添加CAMUS数据集
- [ ] 添加norm、数据增强等数据转换过程
- [ ] 清晰的可视化图

*重要*

*紧急*
- [ ] 写AOT、DeAOT的head
- [ ] 完成半监督的forward,即重写encoder_decoder.py
- [ ] 测试是否能可视化结果
s

## NEXT 
- PSPNet与PIDNet在echovideo做全监督对比
  - 对比分割性能、训练时间、参数量、占用显存
- 全监督与半监督对比
  - sup&semisup，performance是否有提升
- 在哪里可以改进？
  - long-term
  - 时间帧的pos emb
- 不用memories
## 下一个运行
CUDA_VISIBLE_DEVICES=3 python tools/train.py echo/configs/echovideo/pidnet-s_gpm_50ep_echovideo-10.py

CUDA_VISIBLE_DEVICES=3 python tools/test.py work_dirs/pidnet-s_multigpm_200ep_echovideo-10/pidnet-s_multigpm_200ep_echovideo-10.py work_dirs/pidnet-s_multigpm_200ep_echovideo-10/epoch_200.pth

python tools/test.py work_dirs/pidnet-s_gpm_50ep_echovideo-2/pidnet-s_gpm_50ep_echovideo-2.py work_dirs/pidnet-s_gpm_50ep_echovideo-2/epoch_50.pth

## preprocess echonet
```
python echo/tools/preprocess_echonet.py -i /data/dengxiaolong/EchoNet-Dynamic/ -o /data/dengxiaolong/mmseg/echonet1
```

## train
CUDA_VISIBLE_DEVICES=2;python tools/train.py echo/configs/echovideo/pidnet-s_multigpm_200ep_echovideo-10.py

CUDA_VISIBLE_DEVICES=2,3 bash tools/dist_train.sh echo/configs/echovideo/pidnet-s_multigpm_50ep_echovideo-10.py 2

## test
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
work_dirs/pspnet_r50-d8_20k_112x112_echonet/pspnet_r50-d8_20k_112x112_echonet.py \
work_dirs/pspnet_r50-d8_20k_112x112_echonet/iter_20000.pth
python tools/test.py \
work_dirs/pspnet_r50-d8_200epoch_112x112_echocycle/pspnet_r50-d8_200epoch_112x112_echocycle.py \
work_dirs/pspnet_r50-d8_200epoch_112x112_echocycle/epoch_200.pth

## MMseg tensorboard
tensorboard --logdir work_dirs

