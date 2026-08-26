import torch
import torch.nn as nn


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class ConvDecoderCup(nn.Module):
    def __init__(self, feature_channels, feature_sizes, seg_channel=16, use_batchnorm=True):
        super().__init__()
        if len(feature_channels) != 4 or len(feature_sizes) != 4:
            raise ValueError("FGFNet expects four encoder feature maps.")

        decoder_channels = feature_channels[1:] + [seg_channel]
        self.feature_sizes = feature_sizes
        self.conv_more = Conv2dReLU(
            feature_channels[0],
            decoder_channels[0],
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        blocks = []
        for i in range(4):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1] if i < 3 else decoder_channels[i]
            skip_ch = feature_channels[i + 1] if i < 3 else 0
            blocks.append(DecoderBlock(in_ch, out_ch, skip_ch, use_batchnorm))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features):
        if len(features) != 4:
            raise ValueError("Decoder input must contain four feature maps.")

        x = self.conv_more(features[0])
        for i, decoder_block in enumerate(self.blocks):
            skip = features[i + 1] if i < 3 else None
            x = decoder_block(x, skip=skip)
        return x
