# model settings
in_channels = 1
base_channels = 32
expansion_settings = {18: 1, 34: 1, 50: 4, 101: 4, 152: 4, 200: 4}
depth = 50
expansion = expansion_settings[depth]
num_stages = 4
share_weights = True

model = dict(
    type='MirroredXNet',
    act_hidden=False,
    pretrained=None,
    encoder=dict(
        type='AnisResnet',
        depth=depth,
        in_channels=in_channels,
        num_classes=2,
        stem_channels=base_channels,
        base_channels=base_channels,
        num_stages=num_stages,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        deep_stem=True,
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        zero_init_residual=True,
        share_weights=share_weights),
    neck=dict(
        type='NegativeDictionary',
        in_channels=base_channels * expansion * (2 ** (num_stages - 1)),
        num_elements=10,
        threshold=0.625,
        share_dict=True,
        groups=num_stages + 2,
        drop_dict_rate=0.2,
        drop_feat_rate=0.),
    decoder=dict(
        type='AnisBasicDecoder',
        in_channels=base_channels * expansion * (2 ** (num_stages - 1)),
        out_channels=in_channels,
        expansion=expansion,
        num_convs=1,
        num_stages=num_stages,
        concat_feature=False,
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        share_weights=share_weights))
