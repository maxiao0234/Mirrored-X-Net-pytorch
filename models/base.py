import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from timm.models.layers import trunc_normal_
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint


class MirroredXNet(nn.Module, metaclass=ABCMeta):
    """Base class for Mirrored X-Net.

    Args:
        encoder (nn.Module): Backbone encoders of anchor path and contrast
            path, sharing parameters.
        neck (nn.Module): Contrastive learning module for hidden layers
            of encoders. If value is not None, anchor path should not be None.
            Else, skip to decoders Default: None.
        decoder (nn.Module): Backbone decoders of anchor path and contrast
            path, sharing parameters. If value is not None, output auto-encoder
            results. Else, only output encoder results. Default: None.
        pretrained: TODO: Load pretrained parameters for modules. Default: None
    """

    def __init__(self,
                 encoder,
                 neck=None,
                 decoder=None,
                 act_hidden=True,
                 pretrained=None):
        super().__init__()
        assert encoder is not None
        self.encoder = encoder
        self.neck = neck
        self.decoder = decoder
        self.act_hidden = act_hidden
        self.pretrained = pretrained

        # init weights
        self.apply(self._init_weights)

    def build_encoder(self, config):
        """Placeholder for build encoder.
        TODO: build from config.
        """
        pass

    def build_neck(self, config):
        """Placeholder for build neck.
        TODO: build from config.
        """
        pass

    def build_decoder(self, config):
        """Placeholder for build decoder.
        TODO: build from config.
        """
        pass

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        B, C, H, W = x.shape
        self.size = (H, W)
        out_class = self.encoder(x)
        feature = self.encoder.feature
        return feature, out_class

    def forward_neck(self, feature, is_positive):
        if self.act_hidden:
            act_weights = self.encoder.class_head.weight[1, :]
            feature = self.neck(feature, is_positive=is_positive, act_weights=act_weights)
        else:
            feature = self.neck(feature, is_positive=is_positive)
        apm = self.neck.apm
        if isinstance(apm, list):
            apm_list = []
            for apm_ in apm:
                apm_ = F.interpolate(apm_, size=self.size, mode='nearest')
                apm_list.append(apm_)
            apm = torch.cat(apm_list, dim=1)
        apm = torch.mean(apm, dim=1)
        return feature, apm

    def forward_decoder(self, feature):
        out_decoder = self.decoder(feature)
        return out_decoder

    def forward_train(self, anchor, contrast=None):
        if self.neck is not None:
            assert anchor is not None
        feature_a, out_class_a = self.forward_encoder(anchor)
        self.out_class_anchor = out_class_a
        if self.neck is not None:
            feature_a, apm_a = self.forward_neck(feature_a, False)
            self.apm_anchor = apm_a
        if self.decoder is not None:
            out_decoder_a = self.forward_decoder(feature_a)
            self.out_decoder_anchor = out_decoder_a

        if contrast is not None:
            feature_c, out_class_c = self.forward_encoder(contrast)
            self.out_class_contrast = out_class_c
            if self.neck is not None:
                feature_c, apm_c = self.forward_neck(feature_c, True)
                self.apm_contrast = apm_c
            if self.decoder is not None:
                out_decoder_c = self.forward_decoder(feature_c)
                self.out_decoder_contrast = out_decoder_c

    def forward_test(self, x):
        feature, out_class = self.forward_encoder(x)
        self.out_class = out_class
        if self.neck is not None:
            _, apm = self.forward_neck(feature, False)
            self.apm = apm

    def forward(self, x):
        self.forward_test(x)
        return self.apm



