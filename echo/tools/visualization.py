import argparse
import torch
import mmcv
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.visualization import SegLocalVisualizer

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--save_dir', default="./work_dirs/test/", help='the path save visualization')
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


def draw_sem_seg_sum(seg_local_visualizer, image: np.ndarray, gt_sem_seg,
                     pred_sem_seg, classes, gt_palette,
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

    return seg_local_visualizer.get_image()


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
    gt_palette = [[255, 255, 255], [35, 143, 34]]
    pred_palette = [[255, 255, 255], [178, 48, 0]]
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

            ori_imgs, sum_imgs= [], []
            imgs_name = outputs[0].get('img_path').split('/')[-1].split(
                '.')[0] + ".png"
            
            for data_sample in outputs:
                image = data_sample.get('ori_img').data.transpose(1, 2, 0)

                if 'gt_sem_seg' in data_sample:
                    sum_image = draw_sem_seg_sum(
                        seg_local_visualizer,
                        image,
                        data_sample.gt_sem_seg,
                        data_sample.pred_sem_seg,
                        classes,
                        gt_palette,
                        pred_palette)

                else:
                    sum_image = draw_sem_seg(
                        seg_local_visualizer,
                        image,
                        data_sample.pred_sem_seg,
                        classes,
                        pred_palette
                    )

                seg_local_visualizer.set_image(image)
                ori_image = seg_local_visualizer.get_image()

                ori_imgs.append(ori_image)
                sum_imgs.append(sum_image)

            ori_imgs = np.concatenate(ori_imgs, axis=1)
            sum_imgs = np.concatenate(sum_imgs, axis=1)
            drawn_img = np.concatenate((ori_imgs, sum_imgs), axis=0)
            mmcv.imwrite(mmcv.bgr2rgb(drawn_img), save_dir + imgs_name)

        
if __name__ == '__main__':
    main()
