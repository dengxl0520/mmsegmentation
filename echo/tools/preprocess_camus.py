import argparse
import os
import sys
import json
import SimpleITK as sitk
import cv2
import numpy as np
import configparser
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='/data/dengxiaolong/camus/training')
    parser.add_argument('-o', '--output_dir', type=str, default='/data/dengxiaolong/mmseg/camus')
    parser.add_argument('-f', '--split_file', type=str, default='echo/tools/train_val_test.json')
    args = parser.parse_args()

    return args

def generate_list(begin=1, end=450):
    number_list = ['patient'+str(i).zfill(4) for i in range(begin,end+1)]
    return number_list

def read_cfg(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    data = {}

    for line in lines:
        line = line.strip()
        key, value = line.split(":")
        key = key.strip()
        value = value.strip()
        data[key] = value

    return data

def split(data:list, ratios:list):
    total_length = len(data)
    split_points = [int(ratio * total_length) for ratio in ratios]
    random.seed(0)
    random.shuffle(data)
    splits = []
    start = 0
    for point in split_points:
        splits.append(data[start:start+point])
        start += point
    return splits

def preprocess_data(input_path, output_path, split_file):
    # hyperparam
    resize_size = (320,320)
    # generate video name
    patient_num_list = generate_list()
    video_name_list = []
    for patient_num in patient_num_list:
        video_name_list.append(patient_num + '_2CH')
        video_name_list.append(patient_num + '_4CH')

    # split dataset
    if split_file:
        # from split file
        with open(split_file, 'r') as f:
            data = json.load(f)
            train_split, val_split, test_split = data['train'],data['val'],data['test']
    else:
        # random split
        data_split = split(data=video_name_list, ratios=[0.7,0.1,0.2])
        train_split, val_split, test_split = data_split

    # create folder
    if not os.path.exists(output_path + '/videos'):
        os.makedirs(output_path + '/videos/train')
        os.makedirs(output_path + '/videos/val')
        os.makedirs(output_path + '/videos/test')
    if not os.path.exists(output_path + '/annotations'):
        os.makedirs(output_path + '/annotations/train')
        os.makedirs(output_path + '/annotations/val')
        os.makedirs(output_path + '/annotations/test')

    for idx, video_name in enumerate(video_name_list):
        patient_num = video_name.split('_')[0]
        video_mode = video_name.split('_')[1]
        video_path = os.path.join(input_path, patient_num)
        if not os.path.exists(video_path):
            raise ValueError('Directory does not exist')
        
        # read cfg
        cfg = read_cfg(os.path.join(video_path, 'Info_'+ video_mode +'.cfg'))

        # read video
        video = sitk.ReadImage(os.path.join(video_path , video_name + '_sequence.mhd'))
        video_np = sitk.GetArrayFromImage(video)
        # resize video 
        resize_video = []
        for image in video_np:
            resize_image = cv2.resize(image, resize_size)
            resize_video.append(resize_image)
        # to rgb
        resize_video = np.asarray(resize_video)
        resize_video_rgb = np.repeat(resize_video[np.newaxis, :, :, :], 3, axis=0)

        # read annotations
        # assert int(cfg['ED']) < int(cfg['ES'])
        ed = sitk.ReadImage(os.path.join(video_path , video_name + '_ED_gt.mhd'))
        ed_np = sitk.GetArrayFromImage(ed)
        resize_ed = cv2.resize(ed_np[0], resize_size)
        resize_ed[resize_ed != 1] = 0

        es = sitk.ReadImage(os.path.join(video_path , video_name + '_ES_gt.mhd'))
        es_np = sitk.GetArrayFromImage(es)
        resize_es = cv2.resize(es_np[0], resize_size)
        resize_es[resize_es != 1] = 0

        frame_pairs_mask = {
            str(int(cfg['ED']) -1): resize_ed,
            str(int(cfg['ES']) -1): resize_es
        }
            
        # save
        if video_name in train_split:
            video_save_path = os.path.join(output_path, 'videos/train')
            anno_save_path = os.path.join(output_path, 'annotations/train')
        if video_name in val_split:
            video_save_path = os.path.join(output_path, 'videos/val')
            anno_save_path = os.path.join(output_path, 'annotations/val')
        if video_name in test_split:
            video_save_path = os.path.join(output_path, 'videos/test')
            anno_save_path = os.path.join(output_path, 'annotations/test')

        # save video
        np.save(os.path.join(video_save_path, video_name + '.npy'), resize_video_rgb)
        # save anno
        np.savez(
            os.path.join(anno_save_path, video_name + '.npz'),
            fnum_mask=frame_pairs_mask,
            ef=float(cfg['LVef']),
            vol1=float(cfg['LVedv']),
            vol2=float(cfg['LVesv']),
        )
        print(idx+1, video_name)
    
    # creaet split txt
    output_file_train = open(output_path + '/camus_train_filenames.txt', 'w')
    output_file_val = open(output_path + '/camus_val_filenames.txt', 'w')
    output_file_test = open(output_path + '/camus_test_filenames.txt', 'w')

    for name in train_split:
        output_file_train.write(name + '\n')
    for name in val_split:
        output_file_val.write(name + '\n')
    for name in test_split:
        output_file_test.write(name + '\n')

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.input_dir):
        raise ValueError('Input directory does not exist.')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    preprocess_data(input_path=args.input_dir,
                    output_path=args.output_dir,
                    split_file=args.split_file)
