
from mmseg.registry import DATASETS
from mmengine.dataset import BaseDataset
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class CAMUSVideoDataset(BaseSegDataset):
    '''CAMUS Video Dataset.
    '''
    METAINFO = dict(
        classes=('background','lv'),
        palette=[[255,255,255], [128, 0, 0]])
    
    def __init__(self,
                 img_suffix='.npy',
                 seg_map_suffix='.npz',
                 **kwargs):
        super(CAMUSVideoDataset, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)    
        
