import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

from mmengine.dataset import Compose
from mmseg.apis import init_model


T = 10
threshold = 0.5
dataset = 'echonet'
# before_config_path = 'work_dirs/stechonet_r50_sta_stechohead_20k_echonet/stechonet_r50_sta_stechohead_20k_echonet_none.py'
# before_checkpoint_path = 'work_dirs/stechonet_r50_sta_stechohead_20k_echonet/iter_16000.pth'
after_config_path = 'work_dirs/stechonet_r50_sta_stechohead_20k_echonet/stechonet_r50_sta_stechohead_20k_echonet.py'
after_checkpoint_path = 'work_dirs/stechonet_r50_sta_stechohead_20k_echonet/iter_16000.pth'

videos_folder = 'data/echonet/echocycle/videos/test'
# all videos
# videos_list = os.listdir(videos_folder)
# select videos
folder = 'work_dirs/imgs/select'
videos_list = os.listdir(folder)
videos_list = [videos_name.split('.')[0] + '.npy' for videos_name in videos_list]

save_dir = 'work_dirs/aplot'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def set_model_info(models):
    for model in models:
        if model.with_neck:
            model.neck.batchsize = 1
            model.neck.frame_length = T

def get_video_info(video_path: str):
    video = np.load(video_path, allow_pickle=True)
    video = video.swapaxes(0, 1)
    video_info = dict()
    # video_info['img'] = video
    h,w = video.shape[-2:]
    video_info['masks'] = np.zeros((1,h,w))
    video_info['frame_length'] = T
    video_info['video_length'] = video.shape[0]
    return video_info, video

def draw(data, save_path):
    # custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    # sns.set_style("whitegrid", {'grid.alpha': 0.5})
    custom_params = {'grid.alpha': 0.5}
    sns.set_theme(style='whitegrid',font='Times New Roman', font_scale=3, rc=custom_params)
    # sns.set_theme(style="whitegrid")

    # sns.lineplot(data, linewidth=2, palette=["orange","blue"])
    plt.figure(figsize=(10, 5))

    sns.lineplot(data, linewidth=2.5)
    plt.legend().set_visible(False)

    plt.savefig(save_path)
    plt.clf()

def preprare_data(video, video_info, model):
    video_info['img'] = video
    pipeline = [model.cfg.test_pipeline[-1]]
    pipeline = Compose(pipeline)
    video_info = pipeline(video_info)
    video_info = model.data_preprocessor(video_info)

    return video_info

def post_preprocess(seg_logits, threshold):
    if dataset == 'echonet':
        size = (112,112)
    elif dataset == 'camus':
        size = (320,320)

    seg_logits = F.interpolate(
        seg_logits, size=size, mode='bilinear', align_corners=False)
    seg_logits = seg_logits.sigmoid()
    seg_pred = seg_logits > threshold
    seg_pred = seg_pred.int()
    pred = torch.stack([x.sum() for x in seg_pred]).cpu().numpy()
    return pred

def main():

    model = init_model(config=after_config_path, checkpoint=after_checkpoint_path)
    set_model_info([model])
    model.eval()

    with torch.no_grad():
        # slide inference
        for video_name in videos_list:
            video_path = os.path.join(videos_folder,video_name)
            video_info, videos = get_video_info(video_path=video_path)
            start = 0
            end = video_info['video_length']
            pred_list = np.array([])
            # dropout last
            while start < end and end - start > T:
                video = videos[start:start+T]
                start = start + T
                data = preprare_data(video, video_info, model)
                results = model._forward(**data)[-1]
                pred = post_preprocess(results, threshold=threshold)
                pred_list = np.concatenate((pred_list, pred),axis=0)
            x = [i for i in range(pred_list.shape[0])]
            data = pd.DataFrame(pred_list,x)
            draw(data, os.path.join(save_dir,video_name.split(".")[0] + '.png'))

    return 

if __name__ == '__main__':
    main()
