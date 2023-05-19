from mmengine.config import Config, DictAction
from mmengine.runner import Runner

# load config
cfg_path = "work_dirs/pidnet-s_10k_echovideo-2/pidnet-s_10k_echovideo-2.py"
ckpt_path = "work_dirs/pidnet-s_10k_echovideo-2/iter_2000.pth"
cfg = Config.fromfile(cfg_path)
cfg.load_from = ckpt_path

# build the runner from config
runner = Runner.from_cfg(cfg)

# test 
runner._test_loop = runner.build_test_loop(runner._test_loop)
runner.call_hook('before_run')
runner.load_or_resume()
# run 
runner.call_hook('before_val_epoch')
runner.model.eval()
for idx, data_batch in enumerate(runner.test_loop.dataloader):
    runner.test_loop.run_iter(idx,data_batch)
# metrics
for metric in runner.test_loop.evaluator.metrics:
    metric.evaluate(len(metric.results))
