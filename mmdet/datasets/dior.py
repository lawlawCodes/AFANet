from mmdet.registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class DIORDataset(XMLDataset):
    """Dataset for DIOR VOCstyle."""

    METAINFO = {
        'classes':
            ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
             'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golffield',
             'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
             'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'),

        # palette is a list of color tuples, which is used for visualization.
        'palette':
            [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
             (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
             (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
             (165, 42, 42), (49, 1, 255), (242, 5, 5), (182, 182, 255),
             (0, 82, 0), (120, 166, 157)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None
