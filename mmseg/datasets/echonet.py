from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class EchonetDataset(BaseSegDataset):
    '''Echonet-Dynamic Dataset.

    '''
    METAINFO = dict(
        classes=('background','lv'),
        palette=[[255, 255, 255], [128, 0, 0]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs):
        super(EchonetDataset, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
