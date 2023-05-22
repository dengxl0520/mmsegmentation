import mmcv
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmseg.visualization import SegLocalVisualizer


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


is_vis = True

# load config
cfg_path = "work_dirs/pidnet-s_20k_echovideo-2/pidnet-s_20k_echovideo-2.py"
ckpt_path = "work_dirs/pidnet-s_20k_echovideo-2/iter_20000.pth"
cfg = Config.fromfile(cfg_path)
cfg.load_from = ckpt_path

# build the runner from config
runner = Runner.from_cfg(cfg)

# test
runner._test_loop = runner.build_test_loop(runner._test_loop)
runner.call_hook('before_run')
runner.load_or_resume()
runner.call_hook('before_val_epoch')
runner.model.eval()

if is_vis:
    save_dir = "./work_dirs/test/"
    seg_local_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')], save_dir=save_dir)
    dataset_meta = runner.test_evaluator.dataset_meta
    classes = dataset_meta['classes']
    gt_palette = [[255, 255, 255], [35, 143, 34]]
    pred_palette = [[255, 255, 255], [178, 48, 0]]
    seg_local_visualizer.dataset_meta = runner.test_evaluator.dataset_meta

for idx, data_batch in enumerate(runner.test_loop.dataloader):
    outputs = runner.model.test_step(data_batch)
    runner.test_loop.evaluator.process(
        data_samples=outputs, data_batch=data_batch)
    if is_vis:
        drawn_imgs = []
        imgs_name = outputs[0].get('img_path').split('/')[-1].split(
            '.')[0] + ".png"
        for data_sample in outputs:
            frame_idx = str(data_sample.get('frame_idx'))
            image = data_sample.get('ori_img').data.transpose(1, 2, 0)
            image_name = data_sample.get('img_path').split('/')[-1].split(
                '.')[0] + "_" + frame_idx + '.png'

            # # gt
            # gt_img_data = image
            # gt_img_data = draw_sem_seg(
            #     seg_local_visualizer=seg_local_visualizer,
            #     image=gt_img_data,
            #     sem_seg=data_sample.gt_sem_seg,
            #     classes=classes,
            #     palette=gt_palette)
            # # pred
            # pred_img_data = image
            # pred_img_data = draw_sem_seg(
            #     seg_local_visualizer=seg_local_visualizer,
            #     image=pred_img_data,
            #     sem_seg=data_sample.pred_sem_seg,
            #     classes=classes,
            #     palette=pred_palette)
            # # ori image
            # seg_local_visualizer.set_image(image)
            # ori_img = seg_local_visualizer.get_image()

            # drawn_imgs.append(np.concatenate((ori_img, gt_img_data, pred_img_data),
            #                            axis=1))

            sum_image = draw_sem_seg_sum(
                seg_local_visualizer,
                image,
                data_sample.gt_sem_seg,
                data_sample.pred_sem_seg,
                classes,
                gt_palette,
                pred_palette)
            
            seg_local_visualizer.set_image(image)
            ori_img = seg_local_visualizer.get_image()

            drawn_imgs.append(np.concatenate((ori_img, sum_image),
                                       axis=1))

        drawn_img = np.concatenate(drawn_imgs, axis=0)
        mmcv.imwrite(mmcv.bgr2rgb(drawn_img), save_dir + imgs_name)

# compute metrics
for metric in runner.test_loop.evaluator.metrics:
    # metric.evaluate(len(metric.results))
    # sum
    print("All frames results")
    metric.compute_metrics(metric.results)
    # ed
    print("ED frames results")
    metric.compute_metrics(metric.results[::2])
    # es
    print("ES frames results")
    metric.compute_metrics(metric.results[1::2])
