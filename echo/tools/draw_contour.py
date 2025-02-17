import argparse
import torch
import numpy as np
import mmcv
import cv2
from mmengine.visualization import Visualizer
from echo.tools.utils_contour import find_contours

def parse_args():
    parser = argparse.ArgumentParser(
        description='draw contour in image')
    parser.add_argument('--image_path', default='data/echonet/images/train/0X406C79DF5EBE13D9_44.png' ,help='the path of image')
    parser.add_argument('--anno_path', default='data/echonet/annotations/train/0X406C79DF5EBE13D9_44.png', help='the path of image annotation')
    # parser.add_argument('color', help='the color of contour')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    image_path = args.image_path
    anno_path = args.anno_path
    # color = args.color
    # if color == 'green':

    image = mmcv.imread(image_path, channel_order='rgb')
    mask = mmcv.imread(anno_path, flag='grayscale')
    contour = find_contours(mask)
    # dilate
    kernel = np.ones((2,2), np.uint8)
    contour = cv2.dilate(contour, kernel, iterations=1)

    # draw
    contour = contour.astype(bool)
    visualizer = Visualizer(image=image, vis_backends=[dict(type='LocalVisBackend')], save_dir='temp_dir')
    visualizer.draw_binary_masks(contour, colors=[[24,226,24]])
    # visualizer.draw_binary_masks(contour, colors='g')
    visualizer.add_image(image_path.split('/')[-1].split('.')[0], visualizer.get_image())



if __name__ == '__main__':
    main()