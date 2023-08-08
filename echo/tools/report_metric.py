import argparse

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--dataset', help='dataset', default='test')
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

    if args.dataset == 'test':
        for idx, data_batch in enumerate(runner.test_loop.dataloader):
            batchsize = runner.test_loop.dataloader.batch_size
            runner.model.batchsize = batchsize
            outputs = runner.model.test_step(data_batch)
            runner.test_loop.evaluator.process(
                data_samples=outputs, data_batch=data_batch)
    elif args.dataset == 'val':
        for idx, data_batch in enumerate(runner.val_loop.dataloader):
            batchsize = runner.val_loop.dataloader.batch_size
            runner.model.batchsize = batchsize
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
    
    def draw_linear_regression_map(data, xname:str, yname:str):
        import seaborn as sns
        sns.set_theme(style="darkgrid")

        # tips = sns.load_dataset("tips")
        g = sns.jointplot(x=xname, y=yname, data=data,
                        kind="reg", truncate=False,
                        xlim=(0, 100), ylim=(0, 100),
                        color="m", height=7)
        g.savefig('1.png')
        return g


if __name__ == '__main__':
    main()
