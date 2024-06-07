import torch
import torch.nn as nn
import torch.nn.functional as F



class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='group'):
        super(PreActBlock, self).__init__()
        if norm == 'group' :
            self.bn1 = nn.GroupNorm(32, in_planes)
        elif norm == 'layer':
            self.bn1 = nn.GroupNorm(1, in_planes)
        else:
            self.bn1 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        if norm == 'group' :
            self.bn2 = nn.GroupNorm(32, planes)
        elif norm == 'layer' :
            self.bn2 = nn.GroupNorm(1, planes)
        else:
            self.bn2 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else out
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='batch'):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else out
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out
    
class PreActResNetFFT(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        return_features=False,
        num_classes=2,
        drop_prob=0.5,
        input_channels=1,
        fc_layer_dim=100,
        norm = 'group',
    ):
        super(PreActResNetFFT, self).__init__()
        self.input_channels = input_channels
        self.fc_layer_dim = fc_layer_dim
        self.norm = norm
        self.in_planes = 64

        if norm == 'group' :
            self.bn_2D = nn.GroupNorm(32, 512)
        else :
            self.bn_2D = nn.BatchNorm2d(512)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 

        self.layer1 = self._make_layer(block, 64 , num_blocks[0], stride=1, norm=self.norm)
        self.layer2 = self._make_layer(block, 128 , num_blocks[1], stride=2, norm = self.norm)
        self.layer3 = self._make_layer(block, 256 , num_blocks[2], stride=2, norm = self.norm)
        self.layer4 = self._make_layer(block, 512 , num_blocks[3], stride=2, norm = self.norm)

        self.return_features = return_features

        self.avgpool = nn.AdaptiveAvgPool2d((int(self.fc_layer_dim**0.5),int(self.fc_layer_dim**0.5)))
        self.init_weights()    

    def init_weights(self) :
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, num_blocks, stride, norm='group'):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_outputs(self, out):
        layer_1_out = self.layer1(out)
        layer_2_out = self.layer2(layer_1_out)
        layer_3_out = self.layer3(layer_2_out)
        layer_4_out = self.layer4(layer_3_out)
        out = self.avgpool(layer_4_out)

        if self.return_features:
            return [layer_1_out, layer_2_out, layer_3_out, layer_4_out]
        else:
            return out

    def forward(self, kspace):
        kspace = self.conv1(kspace)
        out = self.get_outputs(kspace)
        return out
    
def PreActResNet18FFT(input_channels=1, drop_prob=0.5, return_features=False, fc_layer_dim=1, norm='batch'):
    return PreActResNetFFT(
        PreActBlock, [2, 2, 2, 2], drop_prob=drop_prob, return_features=return_features, input_channels=input_channels, fc_layer_dim=fc_layer_dim, norm=norm
    )

def PreActResNet34FFT(input_channels=1, drop_prob=0.5, return_features=False, fc_layer_dim=1, norm='batch'):
    return PreActResNetFFT(
        PreActBlock, [3, 4, 6, 3], drop_prob=drop_prob, return_features=return_features, input_channels=input_channels, fc_layer_dim=fc_layer_dim, norm=norm
    )

def PreActResNet50FFT(input_channels=1, drop_prob=0.5, return_features=False, fc_layer_dim=1, norm='batch'):
    return PreActResNetFFT(
        PreActBottleneck, [3, 4, 6, 3], drop_prob=drop_prob, return_features=return_features, input_channels=input_channels, fc_layer_dim=fc_layer_dim, norm=norm
    )


def PreActResNet101FFT(input_channels=1, drop_prob=0.5, return_features=False, fc_layer_dim=1, norm='batch'):
    return PreActResNetFFT(
        PreActBottleneck, [3, 4, 23, 3], drop_prob=drop_prob, return_features=return_features, input_channels=input_channels, fc_layer_dim=fc_layer_dim, norm=norm
    )