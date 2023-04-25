# Echo

## TODO List:

*一般*
- [ ] 添加CAMUS数据集
- [ ] 添加辅助head的forward（不一定要）
- [ ] 添加norm、数据增强等数据转换过程

*重要*

*紧急*
- [ ] 写各种head的配置，构建baseline(pidnet)
- [ ] AOT、DeAOT
- [ ] 清晰的可视化图
- [ ] 


## NEXT 
结合LSTT与PIDNet，分层传播context、detail和boundary到当前帧

## 下一个运行
CUDA_VISIBLE_DEVICES=2 python tools/train.py echo/configs/others/pidnet-s_1xb1_100epoch_112x112_echocycle.py

## preprocess echonet
```
python echo/tools/preprocess_echonet.py -i /data/dengxiaolong/EchoNet-Dynamic/ -o /data/dengxiaolong/mmseg/echonet1
```

## train
CUDA_VISIBLE_DEVICES=1 python tools/train.py echo/configs/echonet/pidnet-l_200ep_112x112_echonet.py

CUDA_VISIBLE_DEVICES=1,2 python tools/train.py echo/pspnet_r50-d8_112x112_20k_echonet.py 2

## test
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
work_dirs/pspnet_r50-d8_20k_112x112_echonet/pspnet_r50-d8_20k_112x112_echonet.py \
work_dirs/pspnet_r50-d8_20k_112x112_echonet/iter_20000.pth
python tools/test.py \
work_dirs/pspnet_r50-d8_200epoch_112x112_echocycle/pspnet_r50-d8_200epoch_112x112_echocycle.py \
work_dirs/pspnet_r50-d8_200epoch_112x112_echocycle/epoch_200.pth

## MMseg tensorboard
tensorboard --logdir work_dirs

