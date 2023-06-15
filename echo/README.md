# Echo

## TODO List:
*一般*
- [ ] 添加CAMUS数据集
*重要*
- [ ] 折线图：ES的dice差距，与ED的gt与pred差距，按照es的升序
*紧急*
- [ ] pidnet-s_semi、pidnet-s和pidnet-s_gpm对比

## NEXT 
- 在哪里可以改进？
  - long-term 
    - 使用一个融合模块，提升时序性
      - 融合的时候，要考虑到帧间差异
  - 改进无监督损失
    - 一致性损失：mse、kl
    - 帧间平滑性
      - motivation：利用无标注的帧计算
      - method：abs(delta mse) 
      - limitation：
        - 本来就很低，没有优化空间
        - 即使是优化好了，在performance上（Dice）也体现不出来
    - 特征相似度
      - motivation：让特征提取网络学习更好的特征
      - method：计算特征之间的相似度
  - 挖掘困难像素：
    - motivation：
      - 超声心动图噪声大，边缘模糊
    - method：hardness level
  - 注意更小的物体？
    - motivation：
      - ed和es不平衡，差两个点
      - 1) 因为大小不一样，es更小，更容易被忽略
      - 感觉需要通过时间emb来解决，更后的更小
      - 修改一下pag，让更小的物体，得到更大的注意
        - 问题是：小的物体被错误得预测得小还是大？
      - 2) 由于传播的问题，如果第一帧预测得不好，造成误差积累，导致预测得更差
      - 3) 尺度变化的问题：大小在变化
    - [x] 应该是大小不一样，导致计算的dice偏大
  - 背景类计不计算loss？


- 如何使用无标签帧改进分割质量？
  - 常用的半监督方法
    - 对抗学习
    - 对比学习
    - 伪标签
    - 一致性正则化
    - 混合方法
  - loss函数约束
    - 对无标签帧的特征与有标签帧的特征计算相似度约束
    - 对分割出的mask图计算一致性约束
  - 特征聚合
    - 将无标签帧和有标签帧在特征提取阶段聚合
      - 3DConv
    - 将无标签帧和有标签帧的特征使用某种方式聚合



## train
```bash
CUDA_VISIBLE_DEVICES=1 python tools/train.py echo/configs/echovideo/pidnet-s_multigpm_cons_20k_echovideo-10.py

CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh echo/configs/echovideo/pidnet-s_semi_hl_20k_echovideo-10.py 3
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
