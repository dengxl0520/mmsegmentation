
from mmseg.registry import METRICS

from .iou_metric import IoUMetric

@METRICS.register_module()
class echoMetric(IoUMetric):
    def compute_LVEF():
        pass