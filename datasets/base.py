import os
import os.path as osp
import numpy as np
import mmcv
from mmcv.parallel import DataContainer as DC
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines import to_tensor, Resize, LoadImageFromFile
import torchio as tio


@PIPELINES.register_module()
class LoadCubeScanFromBScan(object):
    """Load a OCT cube-scan from a folder of B-scans."""
    def __init__(self,
                 num_frames,
                 ori_shape=None,
                 bscan_suffix='.bmp',
                 to_float32=False,
                 color_type='grayscale',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.num_frames = num_frames
        self.ori_shape = ori_shape
        self.bscan_suffix = bscan_suffix
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img = []
        for i in range(self.num_frames):
            framename = osp.join(filename, str(i + 1) + self.bscan_suffix)
            bscan_bytes = self.file_client.get(framename)
            bscan = mmcv.imfrombytes(
                bscan_bytes, flag=self.color_type, backend=self.imdecode_backend)
            if self.to_float32:
                bscan = bscan.astype(np.float32)
            img.append(bscan)
        img = np.stack(img, axis=2)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img_shape'] = img.shape
        if self.ori_shape is None:
            results['ori_shape'] = img.shape
        else:
            results['ori_shape'] = self.ori_shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0

        if len(img.shape) < 4:
            num_channels = 1
            img = np.expand_dims(img, axis=-1)
            results['img'] = img
        else:
            num_channels = img.shape[3]
        results['img'] = img.transpose((3, 0, 1, 2))

        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(num_frames={self.num_frames},'
        repr_str += f'(bscan_suffix={self.bscan_suffix},'
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadCubeScan(LoadImageFromFile):
    """Load a OCT cube-scan from original file."""
    def __init__(self, ori_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.ori_shape = ori_shape

    def __call__(self, results):
        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img = np.load(filename)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        if self.ori_shape is None:
            results['ori_shape'] = img.shape
        else:
            results['ori_shape'] = self.ori_shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 4 else img.shape[3]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results


@PIPELINES.register_module()
class GAFormatBundle(object):
    """GA formatting bundle. From 3D image to 2D segmentation.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            img = np.ascontiguousarray(img)
            results['img'] = DC(to_tensor(img), stack=True)
        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None,
                                                     ...].astype(np.int64)),
                stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class Resize3Dto2D(Resize):
    """Resize 3d images & 2d segmentation."""

    def __init__(self, if_img=True, **kwargs):
        super().__init__(**kwargs)
        self.if_img = if_img

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.if_img:
            if self.keep_ratio:
                raise 'TODO: keep_ratio'
            else:
                call_resize = tio.Resize(target_shape=self.img_scale[0])
                img = call_resize(results['img'])

            results['img'] = img
            results['img_shape'] = img.shape
            results['pad_shape'] = img.shape
            results['keep_ratio'] = False
        else:
            pass

    def _resize_seg(self, results):
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(
                    results[key], (results['scale'][1], results['scale'][2]), interpolation='nearest')
            results[key] = gt_seg


@PIPELINES.register_module()
class LoadBScan(object):
    """Load a OCT B-scan.
    Args:
        num_frames (int): Number of B-scans
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Option: "color", "grayscale", "unchanged", "color_ignore_orientation",
            "grayscale_ignore_orientation". Defaults to 'grayscale'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Option:
            'cv2', 'turbojpeg', 'pillow', 'tifffile'. Defaults to 'cv2'
    """
    def __init__(self,
                 num_frames,
                 to_float32=False,
                 color_type='grayscale',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.num_frames = num_frames
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']


        img = []
        for i in range(self.num_frames):
            framename = osp.join(filename, str(i + 1) + self.bscan_suffix)
            bscan_bytes = self.file_client.get(framename)
            bscan = mmcv.imfrombytes(
                bscan_bytes, flag=self.color_type, backend=self.imdecode_backend)
            if self.to_float32:
                bscan = bscan.astype(np.float32)
            img.append(bscan)
        img = np.stack(img, axis=2)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img_shape'] = img.shape
        if self.ori_shape is None:
            results['ori_shape'] = img.shape
        else:
            results['ori_shape'] = self.ori_shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0

        if len(img.shape) < 4:
            num_channels = 1
            img = np.expand_dims(img, axis=-1)
        else:
            num_channels = img.shape[3]

        results['img'] = img.transpose((3, 0, 1, 2))
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(num_frames={self.num_frames},'
        repr_str += f'(bscan_suffix={self.bscan_suffix},'
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadCategories(object):
    """Load annotations for categories."""

    def __init__(self):
        pass

    def __call__(self, results):
        results['gt_class'] = results['ann_info']['cls']
        results['seg_fields'].append('gt_class')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'categories'
        return repr_str
