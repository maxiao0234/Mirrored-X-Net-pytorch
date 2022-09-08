import torch
import torch.nn as nn
import torch.nn.functional as F
from .anis_modules import AnisUpsampling, ConvGroup


class AnisBasicDecoder(nn.Module):
    """Base class for Anis-decoder.

    Args:
        in_channels (int): Number of input feature map channels.
        out_channels (int): Number of output feature map channels.
        expansion (int): Expansion of channels corresponding to encoder.
        num_stages (int): Decoder stages, normally 4.
        num_convs (int): Number of convs for each stage.
        concat_feature (bool): Whether concat the input and output of convs
            for each stage.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): Dictionary to construct and config convolution layer.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        act_cfg (dict): Dictionary to construct and config activation layer.
        share_weights (bool): Whether to share conv kernel of each group.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 num_stages=4,
                 num_convs=3,
                 concat_feature=False,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 share_weights=True):
        super().__init__()
        assert num_convs >= 0
        self.num_stages = num_stages
        self.concat_feature = concat_feature
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.share_weights = share_weights

        self.decoder = nn.ModuleList()
        if concat_feature:
            self.conv_cat_feature = nn.ModuleList()

        for i in range(num_stages - 1):
            self.decoder.append(
                self.make_up_block(
                    in_channels=in_channels // 2 ** i,
                    out_channels=in_channels // 2 ** (i + 1),
                    groups=num_stages + 1 - i,
                    num_convs=num_convs))
            if concat_feature:
                self.conv_cat_feature.append(
                    ConvGroup(
                        in_channels=in_channels // 2 ** i,
                        out_channels=in_channels // 2 ** (i + 1),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        share_weights=self.share_weights,
                        groups=num_stages + 1 - i))

        auto_encoder_neck = []
        auto_encoder_neck.append(
            self.make_up_block(
                in_channels=in_channels // 2 ** (num_stages - 1),
                out_channels=(in_channels // 2 ** (num_stages - 1)) // expansion,
                groups=2,
                num_convs=num_convs))
        auto_encoder_neck.append(
            self.make_up_block(
                in_channels=(in_channels // 2 ** (num_stages - 1)) // expansion,
                out_channels=(in_channels // 2 ** (num_stages - 1)) // expansion,
                groups=1,
                num_convs=num_convs))
        self.auto_encoder_neck = nn.Sequential(*auto_encoder_neck)
        self.auto_encoder_head = nn.Conv2d(
            (in_channels // 2 ** (num_stages - 1)) // expansion,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

    def make_up_block(self, in_channels, out_channels, groups, num_convs):
        up_block = []
        up_block.append(AnisUpsampling())
        up_block.append(
            ConvGroup(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                share_weights=self.share_weights,
                groups=groups))
        for j in range(num_convs - 1):
            up_block.append(
                ConvGroup(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    share_weights=self.share_weights,
                    groups=groups))
        return nn.Sequential(*up_block)

    def forward(self, x):
        assert isinstance(x[-1], list)
        assert (len(x[-1]) > 1)
        out = x[self.num_stages - 1]
        for i in range(len(self.decoder)):
            out = self.decoder[i](out)
            if self.concat_feature:
                for j in range(len(out)):
                    out[j] = torch.cat([out[j], x[self.num_stages - 2 - i][j]], dim=1)
                out = self.conv_cat_feature[i](out)

        out = self.auto_encoder_neck(out)
        out = self.auto_encoder_head(out[0])

        return out



