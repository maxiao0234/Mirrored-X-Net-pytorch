_base_ = [
    './base.py'
]

# 5-fold cross validation
fold = 1

data = dict(
    train_ga=dict(split=['Split/{}.txt'.format(i) for i in range(1, 6) if i != fold]),
    train_normal=dict(split=['Split/{}.txt'.format(i) for i in range(1, 6) if i != fold]),
    test=dict(split='Split/{}.txt'.format(fold)))