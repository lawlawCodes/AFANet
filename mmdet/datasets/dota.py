from mmdet.registry import DATASETS
from .xml_style import XMLDataset

@DATASETS.register_module()
class DOTADataset(XMLDataset):
    """Dataset for DOTA VOCstyle."""

    METAINFO = {
        'classes':
            ('small vehicle', 'large vehicle', 'plane', 'storage tank', 'ship',
             'harbor', 'ground track field', 'soccer ball field', 'tennis court', 'swimming pool',
             'baseball diamond', 'roundabout', 'basketball court', 'bridge', 'helicopter'),

        # palette is a list of color tuples, which is used for visualization.
        'palette':
            [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (255, 0, 0),
             (255, 255, 0), (0, 80, 100), (0, 0, 70), (255, 89, 28), (250, 170, 30),
             (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None



