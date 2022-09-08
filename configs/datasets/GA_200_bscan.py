# dataset settings
dataset_type = 'GADatasetBScan'
img_norm_cfg = dict(mean=127, std=63)
num_frames = 200
img_scale = (224, 224)
ann_roi_scale = (200, 200)
ann_scale = (200, 200)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', prob=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=img_scale, keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train_ga=dict(
        type=dataset_type,
        num_frames=num_frames,
        retain_label=1,
        img_dir='CubeScan',
        ann_dir='SegmentationBScan',
        mask_dir='Segmentation2D_',
        pipeline=train_pipeline),
    train_normal=dict(
        type=dataset_type,
        num_frames=num_frames,
        retain_label=0,
        img_dir='CubeScan',
        ann_dir='SegmentationBScan',
        mask_dir='Segmentation2D_',
        pipeline=train_pipeline),
    train_normal_expand=dict(
        data_root=None,
        type=dataset_type,
        num_frames=num_frames,
        retain_label=0,
        img_dir='CubeScan',
        ann_dir=None,
        mask_dir=None,
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        num_frames=num_frames,
        img_dir='CubeScan',
        ann_dir='SegmentationBScan',
        mask_dir='Segmentation2D_',
        pipeline=test_pipeline))


