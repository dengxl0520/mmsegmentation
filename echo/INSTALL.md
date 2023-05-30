# INSTALL
## conda
conda create -n openmmlab python=3.8.16 -y
conda activate openmmlab
### 安装cuda11.7的torch和torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
## mim
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
## mmseg
cd mmsegmentation
pip install -v -e .
## others 
pip install future tensorboard spatial-correlation-sampler