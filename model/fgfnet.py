import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import ConvDecoderCup
from .encoder import ResNetEncoder


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        weight = self.weight
        var, mean = torch.var_mean(weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        weight = (weight - mean) / torch.sqrt(var + 1e-5)
        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return StdConv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias)


class FADFEM(nn.Module):
    def __init__(self, channel, feature_size, norm="backward", group_channels=32):
        super().__init__()
        self.fft_norm = norm
        self.group_channels = group_channels
        self.groups = int(channel / group_channels)
        if channel % group_channels != 0:
            raise ValueError("channel must be divisible by group_channels.")

        self.fftconv_fus = nn.Sequential(
            conv1x1(channel * 3, channel // 2),
            nn.BatchNorm2d(channel // 2),
            nn.GELU(),
            conv1x1(channel // 2, channel * 2),
            nn.Sigmoid(),
        )
        self.freq_trans_conv_x = nn.Sequential(
            conv1x1(channel * 2, channel),
            nn.GELU(),
            conv1x1(channel, channel * 2),
            nn.Sigmoid(),
        )
        self.freq_trans_conv_y = nn.Sequential(
            conv1x1(channel * 2, channel),
            nn.GELU(),
            conv1x1(channel, channel * 2),
            nn.Sigmoid(),
        )

        u = torch.fft.fftfreq(feature_size).reshape(-1, 1) * feature_size
        v = torch.fft.rfftfreq(feature_size).reshape(1, -1) * feature_size
        distance = torch.sqrt(u ** 2 + v ** 2)
        distance_clipped = torch.clamp(distance, min=2.0, max=feature_size)
        mask = torch.ceil(torch.log2(distance_clipped)).long()
        mask[0, 0] = 0
        self.num_bands = int(torch.max(mask).item() + 1)
        mask_onehot = F.one_hot(mask, num_classes=self.num_bands).float().permute(2, 0, 1)
        self.register_buffer("mask_onehot", mask_onehot)

        input_dim = 2 * group_channels * self.num_bands
        hidden_dim = input_dim * 3
        self.band_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.num_bands),
            nn.Sigmoid(),
        )

        self.convxc = conv1x1(channel, channel)
        self.convyc = conv1x1(channel, channel)

    def forward(self, x, y):
        batch, channels, height, width = x.size()
        x_freq = torch.fft.rfft2(self.convxc(x), norm=self.fft_norm)
        y_freq = torch.fft.rfft2(self.convyc(y), norm=self.fft_norm)
        freq_width = x_freq.size(3)

        maskx, masky = torch.chunk(
            self.fftconv_fus(
                torch.cat(
                    [
                        x_freq.real * x_freq.real + x_freq.imag * x_freq.imag,
                        y_freq.real * y_freq.real + y_freq.imag * y_freq.imag,
                        x_freq.real * y_freq.real + x_freq.imag * y_freq.imag,
                    ],
                    dim=1,
                )
            ),
            chunks=2,
            dim=1,
        )
        x_freq = x_freq * maskx + x_freq
        y_freq = y_freq * masky + y_freq

        x_freq = torch.view_as_real(x_freq).permute(0, 1, 4, 2, 3).contiguous().view(batch, 2 * channels, height, freq_width)
        y_freq = torch.view_as_real(y_freq).permute(0, 1, 4, 2, 3).contiguous().view(batch, 2 * channels, height, freq_width)
        x_freq = self.freq_trans_conv_x(x_freq) + x_freq
        y_freq = self.freq_trans_conv_y(y_freq) + y_freq
        x_freq = torch.view_as_complex(x_freq.view(batch, -1, 2, height, freq_width).permute(0, 1, 3, 4, 2).contiguous())
        y_freq = torch.view_as_complex(y_freq.view(batch, -1, 2, height, freq_width).permute(0, 1, 3, 4, 2).contiguous())

        f1, f2 = torch.chunk(
            torch.einsum("bchw,nhw->bcn", torch.abs(torch.cat([x_freq, y_freq], dim=1)), self.mask_onehot),
            chunks=2,
            dim=1,
        )
        freq_gate = torch.cat([f1, f2], dim=-1)
        freq_gate = freq_gate.view(batch * self.groups, self.group_channels, 2 * self.num_bands)
        freq_gate = self.band_mlp(freq_gate.reshape(batch * self.groups, -1))
        freq_gate = freq_gate.unsqueeze(1).expand(-1, self.group_channels, -1)
        freq_gate = freq_gate.reshape(batch, channels, self.num_bands)
        freq_gate = torch.einsum("bcn,nhw->bchw", freq_gate, self.mask_onehot)

        x_freq = x_freq * freq_gate
        y_freq = y_freq * (1 - freq_gate)
        return torch.fft.irfft2(x_freq, s=(height, width), norm=self.fft_norm) + torch.fft.irfft2(
            y_freq,
            s=(height, width),
            norm=self.fft_norm,
        )


class FGFNet(nn.Module):
    def __init__(self, num_class=8, seg_channel=16, input_size=256, pretrained=True):
        super().__init__()
        self.encoder_1 = ResNetEncoder(in_channels=3, pretrained=pretrained)
        self.encoder_2 = ResNetEncoder(in_channels=1, pretrained=pretrained)

        channels, sizes = self.encoder_1.getchannelsize(input_size)
        self.decoder = ConvDecoderCup(feature_channels=channels, feature_sizes=sizes, seg_channel=seg_channel)
        use_channel = channels[::-1]
        use_size = sizes[::-1]
        self.output_size = input_size

        self.fus_layer0 = FADFEM(use_channel[0], use_size[0][0])
        self.fus_layer1 = FADFEM(use_channel[1], use_size[1][0])
        self.fus_layer2 = FADFEM(use_channel[2], use_size[2][0])
        self.fus_layer3 = FADFEM(use_channel[3], use_size[3][0])
        self.seg = nn.Conv2d(seg_channel, num_class, kernel_size=3, padding=1)

    def forward(self, input_rgb, input_sar):
        x0, x1, x2, x3 = self.encoder_1(input_rgb)
        y0, y1, y2, y3 = self.encoder_2(input_sar)
        f0 = self.fus_layer0(x0, y0)
        f1 = self.fus_layer1(x1, y1)
        f2 = self.fus_layer2(x2, y2)
        f3 = self.fus_layer3(x3, y3)
        out = self.decoder([f3, f2, f1, f0])
        logit = self.seg(out)
        return F.interpolate(logit, size=self.output_size, mode="bilinear", align_corners=False)
