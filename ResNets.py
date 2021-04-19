import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

#resnet34 stem network
class ResidualBlock(nn.Module):
    # module：Residual Block
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),  # inplace = True原地操作
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(nn.Module):  # 224x224x3
    # 实现主module:ResNet34
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),  # ->112
            nn.BatchNorm2d(64),  # 112x112x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel_size=3, stride=2, padding=1
        )  # 56x56x64

        # residual block  3,4,6,3
        self.layer1 = self.make_layer(64, 64, 3)  # 56x56x64
        self.layer2 = self.make_layer(64, 128, 4, stride=2)  # 28x28x128
        self.layer3 = self.make_layer(128, 256, 6, stride=2)  # 14x14x256
        self.layer4 = self.make_layer(256, 512, 3, stride=2)  # 7x7x512
        # fc for classifying
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        shortcut = nn.Sequential(  # option B
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  #
            nn.BatchNorm2d(out_ch)
        )
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):  # 224x224x3
        x = self.pre(x)  # 56x56x64
        x = self.layer1(x)  # 56x56x64
        x = self.layer2(x)  # 28x28x128
        x = self.layer3(x)  # 14x14x256
        x = self.layer4(x)  # 7x7x51
        x = F.avg_pool2d(x, 7)  # 1x1x512
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # 1x1
        return x  # bsx12




#3D ResNet

# def generate_model(model_depth, **kwargs):
#     assert model_depth in [10, 18, 34, 50, 101, 152, 200]
#
#     if model_depth == 10:
#         model = MixedResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
#     elif model_depth == 18:
#         model = MixedResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
#     elif model_depth == 34:
#         model = MixedResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
#     elif model_depth == 50:
#         model = MixedResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
#     elif model_depth == 101:
#         model = MixedResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
#     elif model_depth == 152:
#         model = MixedResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
#     elif model_depth == 200:
#         model = MixedResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
#
#     return model

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Resnet3D(ResNet34):

    def __init__(self,
                 num_classes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
    ):
        super(Resnet3D,self).__init__(num_classes)
        block = BasicBlock
        layers =  [3, 4, 6, 3]
        block_inplanes = get_inplanes()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1_3D = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2_3D = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3_3D = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4_3D = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1_3D(x)
        x = self.layer2_3D(x)
        x = self.layer3_3D(x)
        x = self.layer4_3D(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x