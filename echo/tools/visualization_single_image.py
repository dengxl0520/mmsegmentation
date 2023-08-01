import argparse
import torch
import mmcv
import numpy as np
import cv2
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.visualization import SegLocalVisualizer

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--save_dir', default="./work_dirs/atest/", help='the path save visualization')
    args = parser.parse_args()

    return args


def draw_sem_seg(seg_local_visualizer, image: np.ndarray, sem_seg, classes,
                 palette) -> np.ndarray:
    num_classes = len(classes)

    sem_seg = sem_seg.cpu().data
    ids = np.unique(sem_seg)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    colors = [palette[label] for label in labels]

    seg_local_visualizer.set_image(image)

    # draw semantic masks
    for label, color in zip(labels, colors):
        if label != 0:
            seg_local_visualizer.draw_binary_masks(
                sem_seg == label,
                colors=[color],
                alphas=seg_local_visualizer.alpha)

    return seg_local_visualizer.get_image()

def draw_sem_seg_overlap(seg_local_visualizer, image: np.ndarray, gt_sem_seg,
                     pred_sem_seg, classes, palette):
    num_classes = (len(classes)-1) * 3
    gt_sem_seg = gt_sem_seg.cpu().data
    pred_sem_seg = pred_sem_seg.cpu().data.type(torch.int64)
    sum_seg = 2 * pred_sem_seg + gt_sem_seg
    sum_ids = np.unique(sum_seg)[::-1]
    sum_legal_indices = sum_ids < num_classes
    sum_labels = sum_ids[sum_legal_indices]
    sum_labels = np.array(sum_ids, dtype=np.int64)
    sum_colors = [palette[label] for label in sum_labels]

    seg_local_visualizer.set_image(image)
    
    for label, color in zip(sum_labels[::-1], sum_colors[::-1]):
        if label != 0:
            seg_local_visualizer.draw_binary_masks(
                sum_seg == label,
                colors=[color],
                alphas=seg_local_visualizer.alpha)

    return seg_local_visualizer.get_image()

def draw_sem_seg_sum(seg_local_visualizer, image: np.ndarray, gt_sem_seg,
                     pred_sem_seg, classes, gt_palette, overlap_palette,
                     pred_palette) -> np.ndarray:
    num_classes = len(classes)
    # gt
    gt_sem_seg = gt_sem_seg.cpu().data
    gt_ids = np.unique(gt_sem_seg)[::-1]
    gt_legal_indices = gt_ids < num_classes
    gt_ids = gt_ids[gt_legal_indices]
    gt_labels = np.array(gt_ids, dtype=np.int64)
    gt_colors = [gt_palette[label] for label in gt_labels]

    # pred
    pred_sem_seg = pred_sem_seg.cpu().data
    pred_ids = np.unique(pred_sem_seg)[::-1]
    pred_legal_indices = pred_ids < num_classes
    pred_ids = pred_ids[pred_legal_indices]
    pred_labels = np.array(gt_ids, dtype=np.int64)
    pred_colors = [pred_palette[label] for label in pred_labels]

    # overlap
    overlap_seg = gt_sem_seg * pred_sem_seg
    overlap_ids = np.unique(overlap_seg)[::-1]
    overlap_legal_indices = overlap_ids < num_classes
    overlap_ids = overlap_ids[overlap_legal_indices]
    overlap_labels = np.array(overlap_ids, dtype=np.int64)
    overlap_colors = [overlap_palette[label] for label in overlap_labels]

    seg_local_visualizer.set_image(image)

    # draw pred semantic masks
    for label, color in zip(gt_labels, gt_colors):
        if label != 0:
            seg_local_visualizer.draw_binary_masks(
                gt_sem_seg == label,
                colors=[color],
                alphas=seg_local_visualizer.alpha)

    # draw pred semantic masks
    for label, color in zip(pred_labels, pred_colors):
        if label != 0:
            seg_local_visualizer.draw_binary_masks(
                pred_sem_seg == label,
                colors=[color],
                alphas=seg_local_visualizer.alpha)
            
    # draw overlap semantic masks
    for label, color in zip(overlap_labels, overlap_colors):
        if label != 0:
            seg_local_visualizer.draw_binary_masks(
                overlap_seg == label,
                colors=[color],
                alphas=seg_local_visualizer.alpha)

    return seg_local_visualizer.get_image()

def draw_sem_seg(image, gt_sem_seg, pred_sem_seg, palette):
    '''
        image: [3,h,w]
        gt_sem_seg: []
        pred_sem_seg: []
        palette: [bg, gt, pred, overlap]
    '''
    gt_sem_seg = gt_sem_seg.cpu().data
    pred_sem_seg = pred_sem_seg.cpu().data.type(torch.int64)
    mask = 2 * pred_sem_seg + gt_sem_seg
    mask = mask.squeeze()
    
    ids = np.unique(mask)
    color_mask = np.zeros_like(image)
    for idx in ids:
        color_mask[0][mask == idx] = palette[idx][0]
        color_mask[1][mask == idx] = palette[idx][1]
        color_mask[2][mask == idx] = palette[idx][2]
    
    results = cv2.addWeighted(image, 0.2, color_mask, 0.8, 0)
    # for i in range(mask.shape[0]):
    #     for j in range(mask.shape[1]):
    #         if mask[i,j] !=0:
    #             image[...,i,j] = results[...,i,j]
    mask = mask.unsqueeze(0).repeat(3,1,1)
    image[mask != 0] = results[mask != 0]
    return image

def main():

    # load config
    args = parse_args()
    save_dir = args.save_dir

    # load config
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    pipeline=[
        dict(type='LoadNpyFile', frame_length=10, label_idxs=[0, 9]),
        dict(type='TestPackSegMultiInputs')
    ]
    cfg.test_dataloader.dataset.pipeline = pipeline

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # test
    runner._test_loop = runner.build_test_loop(runner._test_loop)
    runner.call_hook('before_run')
    runner.load_or_resume()
    runner.call_hook('before_val_epoch')
    runner.model.eval()

    seg_local_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')], save_dir=save_dir)
    dataset_meta = runner.test_evaluator.dataset_meta
    classes = dataset_meta['classes']
    # [black, green, red, yellow]
    # [bg, gt, pred, overlap]
    palette = [[255, 255, 255],[37, 143, 36], [178, 48, 0], [178, 151, 0]]
    seg_local_visualizer.dataset_meta = runner.test_evaluator.dataset_meta

    with torch.no_grad():
        for idx, data_batch in enumerate(runner.test_loop.dataloader):
            model = runner.model 
            data_preprocessor = runner.model.data_preprocessor

            data = data_preprocessor(data_batch)
            # seg_logits = model._semi_forward(**data)
            if runner.model.with_neck:
                runner.model.neck.frame_length = len(data['inputs'])
                runner.model.neck.batchsize = 1
            if runner.model.with_decode_head:
                runner.model.decode_head.frame_length = len(data['inputs'])
                runner.model.decode_head.batchsize = 1

            outputs = model.predict(inputs=data['inputs'], data_samples=data['data_samples'])
            # outputs = model.postprocess_result(seg_logits, data['data_samples'])

            img_name = outputs[0].get('img_path').split('/')[-1].split(
                '.')[0] + ".png"
            pred_name = outputs[0].get('img_path').split('/')[-1].split(
                '.')[0] + "_pred.png"
            for data_sample in outputs:
                image = data_sample.get('ori_img').data

                if 'gt_sem_seg' in data_sample:
                    # ori_img
                    mmcv.imwrite(mmcv.bgr2rgb(image.transpose(1, 2, 0)), save_dir + img_name)
                    # vis
                    results = draw_sem_seg(
                            image,
                            data_sample.gt_sem_seg,
                            data_sample.pred_sem_seg,
                            palette)
                    mmcv.imwrite(mmcv.bgr2rgb(results.transpose(1, 2, 0)), save_dir + pred_name)

if __name__ == '__main__':
    main()
