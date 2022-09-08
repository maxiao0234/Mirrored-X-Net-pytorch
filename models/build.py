import torch
from .anis_resnet import AnisResnet
from .anis_decoder import AnisBasicDecoder
from .contrastive_learning_neck import NegativeDictionary
from .base import MirroredXNet


def build_model(config):
    assert config.encoder is not None
    encoder = build_encoder(config.encoder)
    if config.neck is not None:
        neck = build_neck(config.neck)
    else:
        neck = None
    if config.decoder is not None:
        decoder = build_decoder(config.decoder)
    else:
        decoder = None

    model = MirroredXNet(encoder, neck=neck, decoder=decoder, act_hidden=config.act_hidden)
    return model

def build_encoder(config):
    encoder = AnisResnet(
        depth=config.depth,
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        stem_channels=config.stem_channels,
        base_channels=config.base_channels,
        num_stages=config.num_stages,
        strides=config.strides,
        dilations=config.dilations,
        out_indices=config.out_indices,
        deep_stem=config.deep_stem,
        with_cp=config.with_cp,
        conv_cfg=config.conv_cfg,
        norm_cfg=config.norm_cfg,
        act_cfg=config.act_cfg,
        zero_init_residual=config.zero_init_residual,
        share_weights=config.share_weights)
    return encoder


def build_neck(config):
    neck = NegativeDictionary(
        in_channels=config.in_channels,
        num_elements=config.num_elements,
        threshold=config.threshold,
        share_dict=config.share_dict,
        groups=config.groups,
        drop_dict_rate=config.drop_dict_rate,
        drop_feat_rate=config.drop_feat_rate)
    return neck


def build_decoder(config):
    decoder = AnisBasicDecoder(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        expansion=config.expansion,
        num_convs=config.num_convs,
        num_stages=config.num_stages,
        concat_feature=config.concat_feature,
        with_cp=config.with_cp,
        conv_cfg=config.conv_cfg,
        norm_cfg=config.norm_cfg,
        act_cfg=config.act_cfg,
        share_weights=config.share_weights)
    return decoder

