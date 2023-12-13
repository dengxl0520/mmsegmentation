import argparse
import os
import json
import SimpleITK as sitk
import cv2
import numpy as np
import random
from PIL import Image

SEED = 42
RESIZE_SIZE = (320,320)
SPLIT_RATIOS = [0.7,0.1,0.2]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='/data/dengxiaolong/camus/training')
    parser.add_argument('-o', '--output_dir', type=str, default='/data/dengxiaolong/mmseg/camus_random42')
    parser.add_argument('-f', '--split_file', type=str)
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
    random.seed(SEED)
    random.shuffle(data)
    splits = []
    start = 0
    for point in split_points:
        splits.append(data[start:start+point])
        start += point
    return splits

def filter(x):
    # remove error value
    x = np.where(x != 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(x, connectivity=8)
    max_area_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    x = np.where(labels == max_area_label, 1, 0).astype(np.uint8)
    return x

# from https://aistudio.baidu.com/projectdetail/1915947
def resampleSpacing(sitkImage, newspace=(1,1,1)):
    '''
        newResample = resampleSpacing(sitkImage, newspace=[1,1,1])
    '''
    euler3d = sitk.Euler3DTransform()
    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    #新的X轴的Size = 旧X轴的Size *（原X轴的Spacing / 新设定的Spacing）
    new_size = (int(xsize*xspacing/newspace[0]),int(ysize*yspacing/newspace[1]),int(zsize*zspacing/newspace[2]))
    #如果是对标签进行重采样，模式使用最近邻插值，避免增加不必要的像素值
    sitkImage = sitk.Resample(sitkImage,new_size,euler3d,sitk.sitkNearestNeighbor,origin,newspace,direction)
    return sitkImage

def resampleSize(sitkImage, depth):
    '''
        newsitkImage = resampleSize(sitkImage, depth=DEPTH)
    '''
    #重采样函数
    euler3d = sitk.Euler3DTransform()

    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    new_spacing_z = zspacing/(depth/float(zsize))

    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    #根据新的spacing 计算新的size
    newsize = (xsize,ysize,int(zsize*zspacing/new_spacing_z))
    newspace = (xspacing, yspacing, new_spacing_z)
    sitkImage = sitk.Resample(sitkImage,newsize,euler3d,sitk.sitkNearestNeighbor,origin,newspace,direction)
    return sitkImage

def resampleXYSize(sitkImage, new_xsize, new_ysize):
    '''
        newsitkImage = resampleSize(sitkImage, depth=DEPTH)
    '''
    #重采样函数
    euler3d = sitk.Euler3DTransform()

    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    new_spacing_x = xspacing/(new_xsize/float(xsize))
    new_spacing_y = yspacing/(new_ysize/float(ysize))

    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    #根据新的spacing 计算新的size
    newsize = (new_xsize,new_ysize,zsize)
    newspace = (new_spacing_x, new_spacing_y, zspacing)
    sitkImage = sitk.Resample(sitkImage,newsize,euler3d,sitk.sitkNearestNeighbor,origin,newspace,direction)
    return sitkImage


def preprocess_data(input_path, output_path, split_file):
    # hyperparam
    resize_size = RESIZE_SIZE
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
        data_split = split(data=video_name_list, ratios=SPLIT_RATIOS)
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

        # check ed => es
        assert cfg['ED'] == cfg['NbFrame'] or cfg["ES"] ==cfg['NbFrame'] 
        assert float(cfg['LVedv']) > float(cfg['LVesv'])
    
        # read video
        video = sitk.ReadImage(os.path.join(video_path , video_name + '_sequence.mhd'))
        video = resampleXYSize(video, *resize_size)
        video_np = sitk.GetArrayFromImage(video)
        # to rgb
        resize_video_rgb = np.repeat(video_np[np.newaxis, :, :, :], 3, axis=0)
        if not int(cfg['ED']) -1 < int(cfg['ES']) -1:
            # print("ED:" + cfg['ED'])
            # print("ES:" + cfg['ES'])
            # video reverse
            resize_video_rgb = np.flip(resize_video_rgb, axis=1)
            cfg['ED'], cfg['ES'] = cfg['ES'], cfg['ED']
        # print img
        # for i in range(len(video_np)):
        #     cv2.imwrite('./temp/'+str(i)+'.jpg', video_np[i])
        
        # read annotations
        # assert int(cfg['ED']) < int(cfg['ES'])
        ed = sitk.ReadImage(os.path.join(video_path , video_name + '_ED_gt.mhd'))
        resize_ed = resampleXYSize(ed, *resize_size)
        ed_np = sitk.GetArrayFromImage(resize_ed)[0]
        ed_np[ed_np != 1] = 0
        ed_np = filter(ed_np)

        es = sitk.ReadImage(os.path.join(video_path , video_name + '_ES_gt.mhd'))
        resize_es = resampleXYSize(es, *resize_size)
        es_np = sitk.GetArrayFromImage(resize_es)[0]
        es_np[es_np != 1] = 0
        es_np = filter(es_np)

        # check spacing
        assert video.GetSpacing() == resize_ed.GetSpacing()
        assert video.GetSpacing() == resize_es.GetSpacing()
        x_spacing, y_spacing, z_spacing = video.GetSpacing()

        frame_pairs_mask = {
            str(int(cfg['ED']) -1): ed_np,
            str(int(cfg['ES']) -1): es_np
        }
        # check pixel number
        assert ed_np.sum() > es_np.sum()
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
            edv=float(cfg['LVedv']),
            esv=float(cfg['LVesv']),
            spacing=(x_spacing,y_spacing,z_spacing)
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
