import torch
import mmcv
import os
from mmengine.visualization import Visualizer

from mmseg.datasets.transforms.loadingVideo import load_video_and_mask_file

def check_single():
    filename = '/0X1A0A263B22CCD966'
    imgs_path = 'data/echonet/echocycle/videos/train' + filename + '.npy'
    anno_path = 'data/echonet/echocycle/annotations/train' + filename + '.npz'

    imgs, masks, ef = load_video_and_mask_file(imgs_path,anno_path,10)

    visualizer = Visualizer(
            image=imgs[0].transpose(1,2,0),
            vis_backends=[dict(type='LocalVisBackend')],
            save_dir='temp_dir')
    # ori_img
    visualizer.add_image('0X1A0A263B22CCD966', visualizer.get_image(), step=0)
    # mask
    mask = masks[0].astype(bool)
    visualizer.draw_binary_masks(mask)
    visualizer.add_image('0X1A0A263B22CCD966', visualizer.get_image(), step=1)

    visualizer = Visualizer(
            image=imgs[-1].transpose(1,2,0),
            vis_backends=[dict(type='LocalVisBackend')],
            save_dir='temp_dir')
    # ori_img
    visualizer.add_image('0X1A0A263B22CCD966', visualizer.get_image(), step=2)
    # mask
    mask = masks[-1].astype(bool)
    visualizer.draw_binary_masks(mask)
    visualizer.add_image('0X1A0A263B22CCD966', visualizer.get_image(), step=3)

def check_all_camus():
    videos_path = 'data/camus/videos'
    for root, ds, fs in os.walk(videos_path):
        for f in fs:
            video_path = os.path.join(root, f)
            annotation_path = video_path.replace('videos','annotations').replace('npy','npz')
            imgs, masks, ef, edv, esv, spacing= load_video_and_mask_file(video_path, annotation_path,10)
            assert edv > esv and masks[0].sum() > masks[1].sum()

def check_all_echonet():
    videos_path = 'data/echonet/echocycle/videos'
    for root, ds, fs in os.walk(videos_path):
        for f in fs:
            video_path = os.path.join(root, f)
            annotation_path = video_path.replace('videos','annotations').replace('npy','npz')
            imgs, masks, ef, edv, esv, spacing= load_video_and_mask_file(video_path, annotation_path,10)
            assert edv > esv 
            # assert masks[0].sum() > masks[1].sum() 

check_all_echonet()