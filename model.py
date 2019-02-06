import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable

from math import sqrt


def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)
                                  + 1e-8)


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 padding,
                 kernel_size2=None, padding2=None,
                 pixel_norm=True, spectral_norm=False):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel,
                                            kernel1, padding=pad1),
                                nn.LeakyReLU(0.2),
                                EqualConv2d(out_channel, out_channel,
                                            kernel2, padding=pad2),
                                nn.LeakyReLU(0.2))

    def forward(self, input):
        out = self.conv(input)

        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 1

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3,
                 padding=1, style_dim=512, initial=False):
        super().__init__()

        if initial:
            self.conv1 = ConstantInput(in_channel)

        else:
            self.conv1 = EqualConv2d(in_channel, out_channel, kernel_size, padding=padding)

        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise):
        out = self.conv1(input)
        out = self.adain1(out, style)
        out = self.noise1(out, noise)
        out = self.lrelu1(out)

        out = self.conv2(out)
        out = self.adain2(out, style)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)

        return out


class Generator(nn.Module):
    def __init__(self, code_dim):
        super().__init__()

        self.progression = nn.ModuleList([StyledConvBlock(512, 512, 3, 1, initial=True),
                                          StyledConvBlock(512, 512, 3, 1),
                                          StyledConvBlock(512, 512, 3, 1),
                                          StyledConvBlock(512, 512, 3, 1),
                                          StyledConvBlock(512, 256, 3, 1),
                                          StyledConvBlock(256, 128, 3, 1)])

        self.to_rgb = nn.ModuleList([EqualConv2d(512, 3, 1),
                                     EqualConv2d(512, 3, 1),
                                     EqualConv2d(512, 3, 1),
                                     EqualConv2d(512, 3, 1),
                                     EqualConv2d(256, 3, 1),
                                     EqualConv2d(128, 3, 1)])

    def forward(self, style, noise, step=0, alpha=-1):
        out = noise[0]

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if i > 0 and step > 0:
                upsample = F.upsample(out, scale_factor=2)
                out = conv(upsample, style, noise[i])

            else:
                out = conv(out, style, noise[i])

            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](upsample)
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        return out


class StyledGenerator(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        super().__init__()

        self.generator = Generator(code_dim)

        layers = []
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(self, input, noise=None, step=0, alpha=-1, mean_style=None, style_weight=0):
        batch = input.shape[0]

        if noise is None:
            noise = []

            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(torch.randn(batch, 1, size, size, device=input.device))

        style = self.style(input)

        if mean_style is not None:
            style = mean_style + style_weight * (style - mean_style)

        return self.generator(style, noise, step, alpha)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdim=True)

        return style


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.progression = nn.ModuleList([ConvBlock(128, 256, 3, 1),
                                          ConvBlock(256, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(513, 512, 3, 1, 4, 0)])

        self.from_rgb = nn.ModuleList([EqualConv2d(3, 128, 1),
                                       EqualConv2d(3, 256, 1),
                                       EqualConv2d(3, 512, 1),
                                       EqualConv2d(3, 512, 1),
                                       EqualConv2d(3, 512, 1),
                                       EqualConv2d(3, 512, 1)])

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(512, 1)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                mean_std = input.std(0).mean()
                mean_std = mean_std.expand(input.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                out = F.avg_pool2d(out, 2)

                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out
