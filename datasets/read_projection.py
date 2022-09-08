import mmcv
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.builder import DATASETS


@DATASETS.register_module()
class GADatasetProjection(CustomDataset):
    """GA dataset for C-scan/en-face projection.
    File structure is as followed.

        ├── dataset
        │   ├── Projection
        │   │   ├── eye1_cube_z.bmp
        │   │   ├── eye2_cube_z.bmp
        │   │   ├── eye3_cube_z.bmp
        │   ├── Segmentation2D
        │   │   ├── eye1_cube_z.bmp
        │   │   ├── eye2_cube_z.bmp
        │   │   ├── eye3_cube_z.bmp
        │   ├── statistics.xls
    """

    CLASSES = ('geographic_atrophy', 'background')

    PALETTE = [[255, 0, 0], [0, 0, 0]]

    def __init__(self, img_suffix='_cube_z.bmp', seg_map_suffix='_cube_z.bmp', cls=None, **kwargs):
        super(GADatasetProjection, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
        self.cls = cls

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                img_info['ann'] = dict()
                if self.cls is not None:
                    img_info['ann']['cls'] = self.cls
                if ann_dir is not None:
                    img_info['ann']['seg_map'] = img_name + seg_map_suffix
                img_infos.append(img_info)
        else:
            raise 'Dataset Split is None'

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos














