import os.path as osp
import cv2
import mmcv
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class GADatasetBScan(CustomDataset):
    """GA dataset for B-scan. File structure is as followed.

        ├── dataset
        │   ├── CubeScan
        │   │   ├── eye1_cube_z
        │   │   │   ├── 1.bmp
        │   │   │   ├── 2.bmp
        │   │   │   ├── 3.bmp
        │   │   ├── eye2_cube_z
        │   │   ├── eye3_cube_z
        │   ├── SegmentationBScan
        │   │   ├── eye1_cube_z
        │   │   │   ├── 1.bmp
        │   │   │   ├── 2.bmp
        │   │   │   ├── 3.bmp
        │   │   ├── eye2_cube_z
        │   │   ├── eye3_cube_z
        │   ├── Segmentation2D
        │   │   ├── eye1_cube_z.bmp
        │   │   ├── eye2_cube_z.bmp
        │   │   ├── eye3_cube_z.bmp
        │   ├── statistics.xls
    """

    CLASSES = ('geographic_atrophy', 'background')

    PALETTE = [[255, 0, 0], [0, 0, 0]]

    def __init__(self, num_frames, mask_dir=None, mask_suffix='_cube_z.bmp', retain_label=None,
                 img_suffix='_cube_z', seg_map_suffix='_cube_z', **kwargs):
        self.num_frames = num_frames
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.retain_label = retain_label

        super(GADatasetBScan, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                if self.mask_dir is not None:
                    mask_path = osp.join(self.data_root, self.mask_dir, img_name + self.mask_suffix)
                    img_mask = cv2.imread(mask_path)
                else:
                    img_mask = None
                for frame_id in range(self.num_frames):
                    img_info = dict(filename=img_name + img_suffix + '/{}.bmp'.format(frame_id + 1))
                    if img_mask is not None and img_mask[:, frame_id].mean() > 0:
                        img_info['ann'] = dict(cls=1)
                    else:
                        img_info['ann'] = dict(cls=0)
                    if ann_dir is not None:
                        img_info['ann']['seg_map'] = img_name + seg_map_suffix + '/{}.bmp'.format(frame_id + 1)
                    if self.retain_label is not None:
                        if img_info['ann']['cls'] == self.retain_label:
                            img_infos.append(img_info)
                    else:
                        img_infos.append(img_info)
        else:
            raise 'Dataset Split is None'

        # print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos