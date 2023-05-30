# Echo

## TODO List:
*一般*
- [ ] 添加CAMUS数据集
*重要*

*紧急*
- [ ] pidnet-s_semi、pidnet-s和pidnet-s_gpm对比

## NEXT 
- 在哪里可以改进？
  - long-term
  - 改进时空一致性损失
    - 一致性损失：mse、kl
  - 挖掘困难像素：边缘模糊
  - 注意更小的物体？
    - motivation：
    - ed和es不平衡，差两个点
    - 因为大小不一样，es更小，更容易被忽略
    - 感觉需要通过时间emb来解决，更后的更小
    - 

## train
```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py echo/configs/echovideo/pidnet-s_multigpm_cons_20k_echovideo-10.py

CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh echo/configs/echovideo/pidnet-s_semi_cons_mse_20k_echovideo-10.py 3
```
## test
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py
```

### MMseg hook
#### vis hook
```Python
# vis
default_hooks = dict(
    visualization=dict(
      type='SegNpyVisualizationHook', 
      draw=True, 
      interval=50)
)
```
### tools
#### preprocess echonet
```bash
python echo/tools/preprocess_echonet.py -i /data/dengxiaolong/EchoNet-Dynamic/ -o /data/dengxiaolong/mmseg/echonet1
```
#### tensorboard
```bash
conda activate openmmlab
tensorboard --logdir=work_dirs --port=6035 --bind_all 
```
#### report metric
```bash
CUDA_VISIBLE_DEVICES=3 python echo/tools/report_metric.py  \
work_dirs/pidnet-s_gpm_ConsL_20k_echovideo-10/pidnet-s_gpm_ConsL_20k_echovideo-10.py \
work_dirs/pidnet-s_gpm_ConsL_20k_echovideo-10/iter_8000.pth
```
#### visualization
```bash
CUDA_VISIBLE_DEVICES=3 python echo/tools/visualization.py \
work_dirs/pidnet-s_gpm_ConsL_20k_echovideo-10/pidnet-s_gpm_ConsL_20k_echovideo-10.py \
work_dirs/pidnet-s_gpm_ConsL_20k_echovideo-10/iter_10000.pth 
```
