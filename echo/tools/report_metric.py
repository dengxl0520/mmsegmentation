import argparse

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # test
    runner._test_loop = runner.build_test_loop(runner._test_loop)
    runner.call_hook('before_run')
    runner.load_or_resume()
    runner.call_hook('before_val_epoch')
    runner.model.eval()

    for idx, data_batch in enumerate(runner.test_loop.dataloader):
        outputs = runner.model.test_step(data_batch)
        runner.test_loop.evaluator.process(
            data_samples=outputs, data_batch=data_batch)
        
    # compute metrics
    for metric in runner.test_loop.evaluator.metrics:
        # sum
        print("All frames results")
        metric.compute_metrics(metric.results)
        # ed
        print("ED frames results")
        metric.compute_metrics(metric.results[::2])
        # es
        print("ES frames results")
        metric.compute_metrics(metric.results[1::2])


if __name__ == '__main__':
    main()
