from mmdet.registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class NWPUDataset(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
            ('airplane', 'ship', 'storagetank', 'baseball', 'tenniscourt',
             'basketball', 'groundtrackfield', 'harbor', 'bridge', 'vehicle'),

        # palette is a list of color tuples, which is used for visualization.
        'palette':
            [(255, 0, 0), (119, 11, 32), (160, 32, 240), (160, 32, 240), (237, 58, 0),
             (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (0, 255, 0)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None
