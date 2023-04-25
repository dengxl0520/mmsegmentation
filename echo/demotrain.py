from mmengine.runner import Runner
from mmengine.config import Config

cfg = Config.fromfile('echo/config.py')
runner = Runner.from_cfg(cfg)

runner.train()

