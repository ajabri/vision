import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

_REFLECT_PAD = True

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class Identity(nn.Module):
    def __init__(self, size):
        self.size = size
    
    def forward(self, x):
        return x

import numpy as np
class RandomPadder(nn.Module):
    def __init__(self, padding):
        super(RandomPadder, self).__init__()
        if isinstance(padding, int):
            self.padding = [padding] * 4
        else:
            self.padding = padding

    def forward(self, input):
        # mode = np.random.choice(['constant', 'replicate', 'circular', 'reflect'])
        mode = 'reflect'

        if self.padding[0] == 0:
            return input 

        if mode == 'constant':
            return torch.nn.functional.pad(input, self.padding, mode, np.random.random() - 0.5)

        return torch.nn.functional.pad(input, self.padding, mode)

    def extra_repr(self):
        return '{}'.format(self.padding)
        

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, padding=None):
    """3x3 convolution with padding"""
    if padding is None:
        padding = dilation

    if not _REFLECT_PAD:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=padding, groups=groups, bias=False, dilation=dilation)
    else:
        return nn.Sequential(
            RandomPadder(padding),
            # nn.Dropout(p=0.05),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        groups=groups, bias=False, dilation=dilation)
        )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


import torch.nn.functional as F
import utils
class VQ(nn.Module):
    def __init__(self, K=100, temp=0.3):
        super(VQ, self).__init__()
        
        self.temp = temp
        self.K = K

    def forward(self, x):
        l2_x = F.normalize(x, dim=1, p=2)
        some_x = l2_x.flatten(-2)

        # with torch.no_grad():
        #     xxt = F.softmax(torch.einsum('nci,ncj->nij', some_x, some_x), dim=-1)
        #     xent = (-xxt * (xxt + 1e-8).log()).sum(-1)

        #     # xent_idx = xent.sort(dim=-1)[1][:xent.shape[-1]//2]
        #     xent_V, xent_I = xent.sort(dim=-1)
        #     xent_V = F.softmax(xent_V, dim=-1).cpu().detach().numpy()
        #     xent_idx = [np.random.choice(np.arange(xent.shape[-1]), size=(self.K), replace=True, p=xent_V[ii]) for ii in range(xent.shape[0])]
        #     xent_idx = torch.from_numpy(np.stack(xent_idx)).cuda()
        #     # import pdb; pdb.set_trace()

        #     # [:, :self.K]
        #     some_x = some_x.gather(-1, xent_idx[:, None].expand(xent_idx.shape[0], some_x.shape[1], xent_idx.shape[1]))

        #     # import pdb; pdb.set_trace()

        some_idx = torch.randint(some_x.shape[-1], (self.K,))
        some_x = some_x[..., some_idx]

        K = torch.einsum('nchw,nck->nkhw', l2_x, some_x)

        N, k, H, W = K.shape
        K = K.permute(0, 2, 3, 1).flatten(0, -2)
        K = utils.sinkhorn_knopp((K/self.temp).exp(), max_iter=4, tol=10e-5, verbose=False)
        K = K.view(N, H, W, k).permute(0, 3, 1, 2)
        
        x = torch.einsum('nkhw,nck->nchw', K, some_x) #+ x

        return x

    def forward2(self, x):
        from anisotrophic import anisotropic_diffusion
        xout = anisotropic_diffusion(x)
        # import pdb; pdb.set_trace()
        return xout


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, padding=None, residual=True):
        super(BasicBlock, self).__init__()
        # padding = dilation if padding is None else padding

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, padding=padding)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=padding)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.residual = residual

    def forward(self, x):
        identity = x

        # import pdb; pdb.set_trace()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.residual:
            size_diff = np.array(identity.shape) - np.array(out.shape)

            if size_diff.sum() > 0:
                s1 = size_diff[-2] //2
                s2 = size_diff[-1] //2
                ss1, ss2 = out.shape[-2:]

                identity = identity[..., s1:s1+ss1, s2:s2+ss2]

            out += identity

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,
                 strides=[None, None, None, None, None, None], # last two: maxpool, conv1
                 paddings=[None, None, None, None, None, None], 
                 vqs=[False, False, False, False],
                 residual=True):

        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if not _REFLECT_PAD:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=strides[-1] or 2, padding=paddings[-1] or 3,
                                   bias=False)
        else:
            self.conv1 = nn.Sequential(
                RandomPadder(padding=paddings[-1] or 3),
                nn.Conv2d(3, self.inplanes, kernel_size=7, stride=strides[-1] or 2, bias=False)
            )

        self.residual = residual

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[-2] or 2, padding=paddings[-2] or 1) if not (strides[-2] and strides[-2] <= 1) else None
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0] or 1, padding=paddings[0], vq=vqs[-1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1] or 2, padding=paddings[1],
                                       dilate=replace_stride_with_dilation[0], vq=vqs[-3])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2] or 1, padding=paddings[2],
                                       dilate=replace_stride_with_dilation[1], vq=vqs[-2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3] or 1, padding=paddings[3],
                                       dilate=replace_stride_with_dilation[2], vq=vqs[-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, padding=None, vq=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation,
                            norm_layer=norm_layer, padding=padding, residual=self.residual))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, padding=padding, residual=self.residual))
        if vq:
            layers.append(VQ())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)

        x = self.layer1(x)
        
        if self.layer2 is not None:
            x = self.layer2(x)

        if self.layer3 is not None:
            x = self.layer3(x)
        
        if self.layer4 is not None:
            x = self.layer4(x)

        if self.avgpool is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        
        if self.fc is not None:
            x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet18vq(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                    vqs=[False, True, True, False],
                   **kwargs) 

def thinnet(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # kwargs['width_per_group'] = 64 * 2

    return _resnet('resnet10', BasicBlock, [1, 1, 0, 0], False, progress,
                    strides=[2, 2, 1, 1, 1, None],
                   **kwargs)

def thinnet_nopad(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 #* 2

    return _resnet('resnet10', BasicBlock, [1, 1, 0, 0], False, progress,
                    paddings=[0, 0, 0, 0, 0, 0], 
                    strides=[2, 1, 1, 1, 1, None],
                    residual=False,
                   **kwargs)

def thinnet2_nopad(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 #* 2

    return _resnet('resnet10', BasicBlock, [1, 1, 2, 0], False, progress,
                    paddings=[0, 0, 0, 0, 0, 0], 
                    strides=[2, 2, 1, 1, 1, None],
                    # vqs=[False, False, True, True],
                    vqs=[False, False, True, False],
                    residual=True,
                   **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                    strides=[1, 2, 2, 2, None, None],
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)