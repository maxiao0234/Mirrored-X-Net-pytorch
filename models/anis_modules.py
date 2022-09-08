import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer
import math


class ConvGroup(nn.Module):
    """Base class for group convolution. This block simplifies the usage of
    group convolution layers. Allow input list including feature maps with
    different shapes, but input channels should be the same. Allow each group
    to share or not parameters. Besides, we add norm layer and activation layer
    in this module.

    Input (list): Feature maps with the same in_channels.
    Output (list): Each group of input produced by the convolution.
    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        padding (int): Zero-padding added to both sides of the input.
        dilation (int): Spacing between kernel elements.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        share_weights (bool): Whether to share conv kernel of each group.
        groups (int): Number of input groups. It should be the same as length
        of input list, if share_weights is False.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 padding: int = 1,
                 dilation: int = 1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 share_weights: bool = True,
                 groups: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.share_weights = share_weights
        self.groups = groups

        if share_weights:
            self.conv_layer = self.build_conv_layer()
        else:
            self.conv_layer = nn.ModuleList([self.build_conv_layer() for i in range(groups)])

    def build_conv_layer(self):
        conv_layer = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        return conv_layer

    def forward(self, x):
        assert isinstance(x, list)
        if not self.share_weights:
            assert len(x) == self.groups
        out = []
        for i in range(len(x)):
            if self.share_weights:
                out.append(self.conv_layer(x[i]))
            else:
                out.append(self.conv_layer[i](x[i]))
        return out


class ActivationGroup(nn.Module):
    """Applies activations over an input list.

    Args:
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
    """

    def __init__(self, act_cfg=dict(type='ReLU')):
        super().__init__()
        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        assert isinstance(x, list)
        out = []
        for i in range(len(x)):
            out.append(self.act(x[i]))
        return out


class AnisMaxPool2d(nn.Module):
    """Applies 2D max poolings over an input list.

    Args:
        kernel_length: the length of the window to take a max over
            for each direction.
        stride: the stride of the window.
        padding: implicit zero padding to be added on each side.
    """

    def __init__(self,
                 kernel_length: int = 2,
                 stride: int = 2,
                 padding: int = 0):
        super().__init__()
        self.down_vertical = nn.MaxPool2d(
            kernel_size=(kernel_length, 1),
            stride=(stride, 1),
            padding=(padding, 0))
        self.down_horizontal = nn.MaxPool2d(
            kernel_size=(1, kernel_length),
            stride=(1, stride),
            padding=(0, padding))

    def forward(self, x):
        assert isinstance(x, list)
        out = []
        for i in range(len(x)):
            down_h = self.down_vertical(x[i])
            down_w = self.down_horizontal(x[i])
            if i > 0:
                down_ = torch.where(out[-1] > down_h, out[-1], down_h)
                out[-1] = down_
            else:
                out.append(down_h)
            out.append(down_w)
        return out


class AnisDownsampling(nn.Module):
    """Base class for anisotropic down sampling. This block enables feature maps
    to be down sampled in vertical and horizontal respectively, and produces two
    corresponding down sampled feature maps. Besides, sampled feature maps of the
    same shape will be merged.

    Input (list): N groups of feature maps.
    Output (list): N + 1 groups of down sampled feature maps.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels produced by the down sampling.
        kernel_length (int): Size of the window for each direction.
        stride (int): Stride of the down sampling.
        padding (int): Zero-padding added to both sides of the input.
        dilation (int): Spacing between kernel elements.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): Config dict for activation layer.
        share_weights (bool): Whether to share conv kernel of each group.
        groups (int): Number of input groups. It should be the same as length
        of input list, if share_weights is False.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_length: int = 3,
                 stride: int = 2,
                 padding: int = 1,
                 dilation: int = 1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 share_weights: bool = True,
                 groups: int = 1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_length = kernel_length
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.share_weights = share_weights
        self.groups = groups

        if share_weights:
            self.down_vertical = self.build_anis_conv('vertical')
            self.down_horizontal = self.build_anis_conv('horizontal')
            self.conv_fusion = self.build_anis_conv('fusion')
        else:
            self.down_vertical = nn.ModuleList([self.build_anis_conv('vertical') for i in range(groups)])
            self.down_horizontal = nn.ModuleList([self.build_anis_conv('horizontal') for i in range(groups)])
            self.conv_fusion = nn.ModuleList([self.build_anis_conv('fusion') for i in range(groups - 1)])

    def build_anis_conv(self, mode):
        assert mode in ['vertical', 'horizontal', 'fusion']
        if mode == 'vertical':
            in_channels = self.in_channels
            kernel_size = (self.kernel_length, 1)
            stride = (self.stride, 1)
            padding = (self.padding, 0)
        elif mode == 'horizontal':
            in_channels = self.in_channels
            kernel_size = (1, self.kernel_length)
            stride = (1, self.stride)
            padding = (0, self.padding)
        else:
            in_channels = self.out_channels * 2
            kernel_size = (1, 1)
            stride = 1
            padding = 0

        conv_layer = ConvModule(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=self.dilation,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        return conv_layer

    def forward(self, x):
        assert isinstance(x, list)
        if not self.share_weights:
            assert len(x) == self.groups
        out = []
        for i in range(len(x)):
            if self.share_weights:
                down_h = self.down_vertical(x[i])
                down_w = self.down_horizontal(x[i])
            else:
                down_h = self.down_vertical[i](x[i])
                down_w = self.down_horizontal[i](x[i])
            if i > 0:
                fusion_layer = torch.cat([out[-1], down_h], dim=1)
                if self.share_weights:
                    out[-1] = self.conv_fusion(fusion_layer)
                else:
                    out[-1] = self.conv_fusion[i - 1](fusion_layer)
            else:
                out.append(down_h)
            out.append(down_w)
        return out


class AnisUpsampling(nn.Module):
    """Base class for anisotropic up sampling. This block enables feature maps
    to be up sampled in vertical and horizontal respectively, and produces two
    corresponding up sampled feature maps, EXCEPT for two boundary feature maps.
    Besides, sampled feature maps of the same shape will be merged. For a specific
    direction, pixels will be separated by stride, extended by mapping_length.
    Take the mean value, when there are multiple pixels extended to the same position.

    Args:
        mapping_length (int): Length of each pixel for each direction.
        stride (int): Stride of the mapping.
    """

    def __init__(self, mapping_length=5, stride=2):
        super().__init__()

        self.mapping_length = mapping_length
        self.stride = stride

    def forward(self, x):
        assert isinstance(x, list)
        assert(len(x) > 1)
        out = []
        for i in range(len(x)):
            B, C, H, W = x[i].shape
            if len(out) > 0:
                up_padding = [x[i]] + [torch.zeros_like(x[i])] * (self.stride - 1)
                up_padding = torch.stack(up_padding, dim=4).view(B, C, H, W * self.stride)
                norm_padding = torch.ones((1, 1, 1, W), dtype=x[i].dtype, device=x[i].device)
                norm_padding = [norm_padding] + [torch.zeros_like(norm_padding)] * (self.stride - 1)
                norm_padding = torch.stack(norm_padding, dim=4).view(1, 1, 1, W * self.stride)

                up_w = [F.pad(up_padding, (i, self.mapping_length - i, 0, 0)) for i in range(self.mapping_length)]
                norm = [F.pad(norm_padding, (i, self.mapping_length - i, 0, 0)) for i in range(self.mapping_length)]

                norm = sum(norm)
                norm = torch.where(norm == torch.zeros_like(norm), torch.ones_like(norm), norm)
                up_w = sum(up_w) / norm
                start_id = math.ceil(self.mapping_length / self.stride) - 1
                up_w = up_w[:, :, :, start_id: start_id + W * self.stride]

                out[-1] = (out[-1] + up_w) / 2

            if len(out) < len(x) - 1:
                up_padding = [x[i]] + [torch.zeros_like(x[i])] * (self.stride - 1)
                up_padding = torch.stack(up_padding, dim=3).view(B, C, H * self.stride, W)
                norm_padding = torch.ones((1, 1, H, 1), dtype=x[i].dtype, device=x[i].device)
                norm_padding = [norm_padding] + [torch.zeros_like(norm_padding)] * (self.stride - 1)
                norm_padding = torch.stack(norm_padding, dim=3).view(1, 1, H * self.stride, 1)

                up_h = [F.pad(up_padding, (0, 0, i, self.mapping_length - i)) for i in range(self.mapping_length)]
                norm = [F.pad(norm_padding, (0, 0, i, self.mapping_length - i)) for i in range(self.mapping_length)]

                norm = sum(norm)
                norm = torch.where(norm == torch.zeros_like(norm), torch.ones_like(norm), norm)
                up_h = sum(up_h) / norm
                start_id = math.ceil(self.mapping_length / self.stride) - 1
                up_h = up_h[:, :, start_id: start_id + H * self.stride, :]

                out.append(up_h)
        return out


class AnisUpsampling2(nn.Module):
    def __init__(self, stride=2.):
        super().__init__()

        self.stride = stride
        self.up_vertical = self.build_anis_upsample('vertical')
        self.up_horizontal = self.build_anis_upsample('horizontal')

    def build_anis_upsample(self, mode):
        assert mode in ['vertical', 'horizontal']
        if mode == 'vertical':
            scale_factor = (self.stride, 1.)
        else:
            scale_factor = (1., self.stride)
        up_layer = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        return up_layer

    def forward(self, x):
        assert isinstance(x, list)
        assert(len(x) > 1)
        out = []
        for i in range(len(x)):
            if len(out) > 0:
                up_w = self.up_horizontal(x[i])
                out[-1] = (out[-1] + up_w) / 2
            if len(out) < len(x) - 1:
                up_h = self.up_vertical(x[i])
                out.append(up_h)
        return out


