import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        self._channels = [256, 512, 1024, 2048]

        old_conv = backbone.conv1
        self.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        if pretrained:
            self._init_conv1(old_conv.weight)

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def _init_conv1(self, old_weight):
        new_weight = self.conv1.weight.data
        if new_weight.shape[1] <= 3:
            new_weight.copy_(old_weight[:, : new_weight.shape[1]])
            return

        repeat = new_weight.shape[1] // 3
        remainder = new_weight.shape[1] % 3
        new_weight[:, : repeat * 3] = old_weight.repeat(1, repeat, 1, 1)
        if remainder > 0:
            new_weight[:, repeat * 3:] = old_weight[:, :remainder]
        new_weight.mul_(3.0 / new_weight.shape[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4

    def getchannelsize(self, size):
        if isinstance(size, int):
            height, width = size, size
        else:
            height, width = size

        sizes = [
            (height // 4, width // 4),
            (height // 8, width // 8),
            (height // 16, width // 16),
            (height // 32, width // 32),
        ]
        return list(self._channels)[::-1], sizes[::-1]
