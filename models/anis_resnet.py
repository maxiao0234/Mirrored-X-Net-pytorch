import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from .anis_modules import AnisDownsampling, ActivationGroup, ConvGroup, AnisMaxPool2d


class AnisBasicBlock(nn.Module):
    """Basic block for Anis-ResNet."""

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 share_weights=True,
                 groups=1):
        super().__init__()

        if stride > 1:
            self.conv1 = AnisDownsampling(
                in_channels=inplanes,
                out_channels=planes,
                kernel_length=5,
                stride=stride,
                padding=2,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                share_weights=share_weights,
                groups=groups)
            groups += 1
        else:
            self.conv1 = ConvGroup(
                in_channels=inplanes,
                out_channels=planes,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                share_weights=share_weights,
                groups=groups
            )

        self.conv2 = ConvGroup(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
            share_weights=share_weights,
            groups=groups
        )

        self.act = ActivationGroup(act_cfg)
        self.downsample = downsample
        self.with_cp = with_cp

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            for i in range(len(out)):
                out[i] += identity[i]

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.act(out)

        return out


class AnisBottleneck(nn.Module):
    """Bottleneck block for Anis-ResNet."""

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 share_weights=True,
                 groups=1):
        super().__init__()

        self.conv1 = ConvGroup(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=dilation,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            share_weights=share_weights,
            groups=groups
        )

        if stride > 1:
            self.conv2 = AnisDownsampling(
                in_channels=planes,
                out_channels=planes,
                kernel_length=5,
                stride=stride,
                padding=2,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                share_weights=share_weights,
                groups=groups)
            groups += 1
        else:
            self.conv2 = ConvGroup(
                in_channels=planes,
                out_channels=planes,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                share_weights=share_weights,
                groups=groups
            )

        self.conv3 = ConvGroup(
            in_channels=planes,
            out_channels=planes * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=dilation,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
            share_weights=share_weights,
            groups=groups
        )

        self.act = ActivationGroup(act_cfg)
        self.downsample = downsample
        self.with_cp = with_cp

    def forward(self, x):
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            for i in range(len(out)):
                out[i] += identity[i]

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.act(out)

        return out


class AnisResnet(nn.Module):
    """Anis-encoder using resnet as template.

    Args:
        depth (int): Number of layers of encoder. Options: [18, 34, 50, 101, 152, 200].
        in_channels (int): Number of input image channels. Default: 1.
        num_classes (int): Number of classes for classification head. Default: 2
        stem_channels (int): Number of stem channels. Default: 64.
        base_channels (int): Number of base channels of res layer. Default: 64.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): Dictionary to construct and config convolution layer.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        act_cfg (dict): Dictionary to construct and config activation layer.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        share_weights (bool): Whether to share conv kernel of each group.
    """

    arch_settings = {
        18: (AnisBasicBlock, (2, 2, 2, 2)),
        34: (AnisBasicBlock, (3, 4, 6, 3)),
        50: (AnisBottleneck, (3, 4, 6, 3)),
        101: (AnisBottleneck, (3, 4, 23, 3)),
        152: (AnisBottleneck, (3, 8, 36, 3)),
        200: (AnisBottleneck, (3, 24, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=1,
                 num_classes=2,
                 stem_channels=64,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 deep_stem=False,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 zero_init_residual=True,
                 share_weights=True):
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.deep_stem = deep_stem
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.share_weights = share_weights
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        self.groups = 3
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = base_channels * 2 ** i
            res_layer = self.make_res_layer(
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * base_channels * 2 ** (
                len(self.stage_blocks) - 1) * 6
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.class_head = nn.Linear(self.feat_dim, self.num_classes, bias=False)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                AnisDownsampling(
                    in_channels=in_channels,
                    out_channels=stem_channels // 2,
                    kernel_length=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    share_weights=self.share_weights,
                    groups=1),
                ConvGroup(
                    in_channels=stem_channels // 2,
                    out_channels=stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    share_weights=self.share_weights,
                    groups=2),
                ConvGroup(
                    in_channels=stem_channels // 2,
                    out_channels=stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    share_weights=self.share_weights,
                    groups=2))
        else:
            self.conv1 = AnisDownsampling(
                in_channels=in_channels,
                out_channels=stem_channels,
                kernel_length=7,
                stride=2,
                padding=3,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                share_weights=self.share_weights,
                groups=1)
        self.maxpool = AnisMaxPool2d(kernel_length=3, stride=2, padding=1)

    def make_res_layer(self, planes, num_blocks, stride, dilation):
        downsample = None

        if self.inplanes != planes * self.block.expansion:
            if stride != 1:
                downsample = AnisDownsampling(
                    in_channels=self.inplanes,
                    out_channels=planes * self.block.expansion,
                    kernel_length=1,
                    stride=stride,
                    padding=0,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    share_weights=self.share_weights,
                    groups=self.groups)
            else:
                downsample = ConvGroup(
                    in_channels=self.inplanes,
                    out_channels=planes * self.block.expansion,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    share_weights=self.share_weights,
                    groups=self.groups)
        layers = []
        layers.append(
            self.block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                share_weights=self.share_weights,
                groups=self.groups))
        if stride > 1:
            self.groups += 1
        self.inplanes = planes * self.block.expansion
        for i in range(1, num_blocks):
            layers.append(
                self.block(
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=1,
                    dilation=dilation,
                    with_cp=self.with_cp,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    share_weights=self.share_weights,
                    groups=self.groups))

        return nn.Sequential(*layers)

    def forward_feature(self, x):
        x = [x]
        outs = []

        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)

        x = self.maxpool(x)

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)

            x = res_layer(x)

            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def forward(self, x):
        self.feature = self.forward_feature(x)
        out = []
        for i in range(len(self.feature[-1])):
            out.append(self.avg_pool(self.feature[-1][i]))
        out = torch.cat(out, dim=1)
        out = self.class_head(torch.flatten(out, 1))
        return out