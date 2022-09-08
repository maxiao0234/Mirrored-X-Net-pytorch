_base_ = [
    '../datasets/GA_200_bscan.py',
    '../models/mxnet.py',
    '../schedules/adamw.py',
]

data_root = '/home/Data/maxiao/GA/GA 200'
expand_root = None

data = dict(
    train_ga=dict(data_root=data_root),
    train_normal=dict(data_root=data_root),
    train_normal_expand=dict(data_root=expand_root),
    test=dict(data_root=data_root))