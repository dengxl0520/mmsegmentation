import torch
from typing import Dict, Optional, Sequence, Union

from mmengine.registry import HOOKS
from mmengine.hooks import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class LossHook(Hook):
    """loss hook.

    """

    def __init__(self, interval=1):
        self.interval = interval

    def before_train_iter(self,
                     runner,
                     batch_idx: int,
                     data_batch: DATA_BATCH = None,
                     mode: str = 'train') -> None:
        current_iters = runner.message_hub.runtime_info['iter'] + 1
        max_iters = runner.message_hub.runtime_info['max_iters']
        t = pow(current_iters / max_iters, 2)
        runner.model.decode_head.loss_decode[4].t = t
