_base_ = [
    '../datasets/GA_128_bscan.py',
    '../models/mxnet.py',
    '../schedules/adamw.py',
]

data_root = '/home/Data/maxiao/GA/GA 128'
expand_root = '/home/Data/maxiao/BSCAN/NORMAL 63'
# expand_root = None

data = dict(
    train_ga=dict(data_root=data_root),
    train_normal=dict(data_root=data_root),
    train_normal_expand=dict(
        data_root=expand_root,
        img_suffix='_cube_raw',
        split='Split/all.txt'),
    test=dict(data_root=data_root))