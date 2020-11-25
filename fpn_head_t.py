import torch.nn as nn
import torch
import numpy as np

# 定义双线性插值，作为转置卷积的初始化权重参数
def bilinear_kernel(in_channels, out_channels, kernel_size):
   factor = (kernel_size + 1) // 2
   if kernel_size % 2 == 1:
       center = factor - 1
   else:
       center = factor - 0.5
   og = np.ogrid[:kernel_size, :kernel_size]
   filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
   weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
   weight[range(in_channels), range(out_channels), :, :] = filt
   return torch.from_numpy(weight)


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, use_2x2_conv=False):

        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True))
        if use_2x2_conv:
            self.conv_t = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, padding=0, stride=2)
            self.conv_t.weight.data = bilinear_kernel(out_channels, out_channels, 2)
        else:
            self.conv_t = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)
            self.conv_t.weight.data = bilinear_kernel(out_channels, out_channels, 3)

    def forward(self, x, size):
        x = self.block(x)
        if self.upsample:
            x = self.conv_t(x)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels, use_2x2_conv=False):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)
        if use_2x2_conv:
            self.conv_t = nn.ConvTranspose2d(pyramid_channels, pyramid_channels, kernel_size=2, padding=0, stride=2)
            self.conv_t.weight.data = bilinear_kernel(pyramid_channels, pyramid_channels, 2)
        else:
            self.conv_t = nn.ConvTranspose2d(pyramid_channels, pyramid_channels, kernel_size=3, padding=1, stride=2)
            self.conv_t.weight.data = bilinear_kernel(pyramid_channels, pyramid_channels, 3)

    def forward(self, x):
        x, skip = x
        x = self.conv_t(x)
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        self.blocks = []
        if n_upsamples==0:
            self.blocks.append(Conv3x3GNReLU(in_channels, out_channels, upsample=False))
        elif n_upsamples==1:
            self.blocks.append(Conv3x3GNReLU(in_channels, out_channels, upsample=True, use_2x2_conv=True))
        else:
            self.blocks.append(Conv3x3GNReLU(in_channels, out_channels, upsample=True))
            for _ in range(1, n_upsamples - 1):
                self.blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))
            self.blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True, use_2x2_conv=True))

        self.blocks_name = []
        for i, block in enumerate(self.blocks):
            self.add_module("Block_{}".format(i), block)
            self.blocks_name.append("Block_{}".format(i))

    def forward(self, x, sizes = []):
        for i, block_name in enumerate(self.blocks_name):
            x = getattr(self, block_name)(x, sizes[i])
        return x


class FPNDecoder(nn.Module):
    def __init__(self, encoder_channels, pyramid_channels=256, segmentation_channels=128,
                 final_upsampling=4, final_channels=1, dropout=0.0):
        super().__init__()
        self.final_upsampling = final_upsampling
        self.conv1 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3], use_2x2_conv=True)

        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=3)
        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=2)
        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=1)
        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0)

        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.final_conv = nn.Conv2d(segmentation_channels, final_channels, kernel_size=5, padding=2)
        self.conv_t = Convt_upsampling(final_channels, final_channels)

    def forward(self, x):
        _, c2, c3, c4, c5 = x

        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])

        s5 = self.s5(p5, sizes=[c4.size()[-2:], c3.size()[-2:], c2.size()[-2:]])
        s4 = self.s4(p4, sizes=[c3.size()[-2:], c2.size()[-2:]])
        s3 = self.s3(p3, sizes=[c2.size()[-2:]])
        s2 = self.s2(p2, sizes=[c2.size()[-2:]])

        x = s5 + s4 + s3 + s2

        x = self.dropout(x)
        x = self.final_conv(x)

        if self.final_upsampling is not None and self.final_upsampling > 1:
            x = self.conv_t(x)
        return x


class Convt_upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_t1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # self.conv_t1.weight.data = bilinear_kernel(in_channels, out_channels, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv_t2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        # self.conv_t2.weight.data = bilinear_kernel(out_channels, out_channels, 2)

    def forward(self, input):
        out = self.conv_t1(input)
        return self.conv_t2(out)