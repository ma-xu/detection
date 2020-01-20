import logging

import torch.nn as nn
import torch
import torch.utils.checkpoint as cp

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from mmdet.ops import DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_norm_layer

from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter


class PRMLayer(nn.Module):
    def __init__(self,groups=64,mode='dotproduct'):
        super(PRMLayer, self).__init__()
        self.mode = mode
        self.groups = groups
        self.max_pool = nn.AdaptiveMaxPool2d(1,return_indices=True)
        self.weight = Parameter(torch.zeros(1,self.groups,1,1))
        self.bias = Parameter(torch.ones(1,self.groups,1,1))
        self.sig = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.one = Parameter(torch.ones(1,self.groups,1))
        self.zero = Parameter(torch.zeros(1, self.groups, 1))
        self.theta = Parameter(torch.rand(1,2,1,1))
        self.scale =  Parameter(torch.ones(1))

    def forward(self, x):

        b,c,h,w = x.size()
        position_mask = self.get_position_mask(x, b, h, w, self.groups)
        # Similarity function
        query_value, query_position = self.get_query_position(x, self.groups)  # shape [b*num,2,1,1]
        query_value = query_value.view(b*self.groups,-1,1)
        x_value = x.view(b*self.groups,-1,h*w)
        similarity_max = self.get_similarity(x_value, query_value, mode=self.mode)
        similarity_gap = self.get_similarity(x_value, self.gap(x).view(b*self.groups,-1,1), mode=self.mode)

        similarity_max = similarity_max.view(b,self.groups,h*w)

        Distance = abs(position_mask - query_position)
        Distance = Distance.type(query_value.type())
        # Distance = torch.exp(-Distance * self.theta)
        distribution = Normal(0, self.scale)
        Distance = distribution.log_prob(Distance * self.theta).exp().clone()
        Distance = (Distance.mean(dim=1)).view(b, self.groups, h * w)
        # # add e^(-x), means closer more important
        # Distance = torch.exp(-Distance * self.theta)
        # Distance = (self.distance_embedding(Distance)).reshape(b, self.groups, h*w)
        similarity_max = similarity_max*Distance


        similarity_gap = similarity_gap.view(b, self.groups, h*w)
        similarity = similarity_max*self.zero+similarity_gap*self.one



        context = similarity - similarity.mean(dim=2, keepdim=True)
        std = context.std(dim=2, keepdim=True) + 1e-5
        context = (context/std).view(b,self.groups,h,w)
        # affine function
        context = context * self.weight + self.bias
        context = context.view(b*self.groups,1,h,w)\
            .expand(b*self.groups, c//self.groups, h, w).reshape(b,c,h,w)
        value = x*self.sig(context)

        return value

    def get_position_mask(self,x,b,h,w,number):
        mask = (x[0, 0, :, :] != 2020).nonzero()
        mask = (mask.reshape(h,w, 2)).permute(2,0,1).expand(b*number,2,h,w)
        return mask


    def get_query_position(self, query,groups):
        b,c,h,w = query.size()
        value = query.view(b*groups,c//groups,h,w)
        sumvalue = value.sum(dim=1,keepdim=True)
        maxvalue,maxposition = self.max_pool(sumvalue)
        t_position = torch.cat((maxposition//w,maxposition % w),dim=1)

        t_value = value[torch.arange(b*groups),:,t_position[:,0,0,0],t_position[:,1,0,0]]
        t_value = t_value.view(b, c, 1, 1)
        return t_value, t_position

    def get_similarity(self,query, key_value, mode='dotproduct'):
        if mode == 'dotproduct':
            similarity = torch.matmul(key_value.permute(0, 2, 1), query).squeeze(dim=1)
        elif mode == 'l1norm':
            similarity = -(abs(query - key_value)).sum(dim=1)
        elif mode == 'gaussian':
            # Gaussian Similarity (No recommanded, too sensitive to noise)
            similarity = torch.exp(torch.matmul(key_value.permute(0, 2, 1), query))
            similarity[similarity == float("Inf")] = 0
            similarity[similarity <= 1e-9] = 1e-9
        else:
            similarity = torch.matmul(key_value.permute(0, 2, 1), query)
        return similarity







def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 normalize=dict(type='BN'),
                 dcn=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, "Not implemented yet."

        self.norm1_name, norm1 = build_norm_layer(normalize, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalize, planes, postfix=2)

        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = conv3x3(planes, planes)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 normalize=dict(type='BN'),
                 dcn=None):
        """Bottleneck block for ResNetSE.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        self.inplanes = inplanes
        self.planes = planes
        self.normalize = normalize
        self.dcn = dcn
        self.with_dcn = dcn is not None
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(normalize, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalize, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            normalize, planes * self.expansion, postfix=3)

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                planes,
                deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation)
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)

        self.prm  = PRMLayer()

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        self.normalize = normalize

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if not self.with_dcn:
                out = self.conv2(out)
            elif self.with_modulated_dcn:
                offset_mask = self.conv2_offset(out)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                out = self.conv2(out, offset, mask)
            else:
                offset = self.conv2_offset(out)
                out = self.conv2(out, offset)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            out = self.prm(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   normalize=dict(type='BN'),
                   dcn=None):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(normalize, planes * block.expansion)[1],
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            with_cp=with_cp,
            normalize=normalize,
            dcn=dcn))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                1,
                dilation,
                style=style,
                with_cp=with_cp,
                normalize=normalize,
                dcn=dcn))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNetPRM(nn.Module):
    """ResNetSE backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        normalize (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 normalize=dict(type='BN', frozen=False),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 with_cp=False,
                 zero_init_residual=True):
        super(ResNetPRM, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.normalize = normalize
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                normalize=normalize,
                dcn=dcn)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self):
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.normalize, 64, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNetPRM, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()



def demo():
    net = ResNetPRM(depth=50)
    net.to('cuda')
    y = net(torch.randn((2, 3, 1333, 800),device=torch.device("cuda")))
    print("SE allocated: {}".format(torch.cuda.memory_allocated()))
    # print(y)

# demo()
