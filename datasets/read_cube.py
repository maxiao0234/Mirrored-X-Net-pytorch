from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class GADatasetCube(CustomDataset):
    """GA dataset for 3D cube-scan. File structure is as followed.

        ├── dataset
        │   ├── CubeScan
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
        │   ├── Projection
        │   │   ├── eye1_cube_z.bmp
        │   │   ├── eye2_cube_z.bmp
        │   │   ├── eye3_cube_z.bmp
        │   ├── statistics.xls
    """

    CLASSES = ('geographic_atrophy', 'background')

    PALETTE = [[255, 0, 0], [0, 0, 0]]

    def __init__(self, **kwargs):

        super(GADatasetCube, self).__init__(
            img_suffix='_cube_z.npy', seg_map_suffix='_cube_z.bmp', **kwargs)


