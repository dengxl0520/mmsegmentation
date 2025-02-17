import argparse
import imageio
import os
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    test_pipeline = cfg.test_dataloader.dataset.pipeline 
    test_pipeline[-1] = dict(type='TestPackSegMultiInputs') # 会污染原来的配置文件,需要手动改回PackSegMultiInputs

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

    for idx, data_batch in enumerate(runner.test_loop.dataloader):
        model = runner.model 
        data_preprocessor = model.data_preprocessor

        data = data_preprocessor(data_batch)
        model.set_train_cfg(data['data_samples'], batchsize=1)
        outputs = model.predict(**data)

        drawn_imgs = []
        imgs_name = outputs[0].get('img_path').split('/')[-1].split(
            '.')[0] + ".gif"
        
        for data_sample in outputs:
            image = data_sample.get('ori_img').data.transpose(1, 2, 0)
            sum_image = draw_sem_seg(
                seg_local_visualizer,
                image,
                data_sample.pred_sem_seg,
                classes,
                pred_palette
            )

            seg_local_visualizer.set_image(image)
            ori_image = seg_local_visualizer.get_image()

            drawn_imgs.append(np.concatenate((ori_image, sum_image),
                                    axis=1))

        imageio.mimsave(save_dir + imgs_name, drawn_imgs, 'GIF', duration = 0.3)
        

        
if __name__ == '__main__':
    main()
