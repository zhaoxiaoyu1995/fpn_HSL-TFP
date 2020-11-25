import torch
import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        # x[:,:,1:,:]-x[:,:,:h_x-1,:]就是对原图进行错位，分成两张像素位置差1的图片，第一张图片
        # 从像素点1开始（原图从0开始），到最后一个像素点，第二张图片从像素点0开始，到倒数第二个
        # 像素点，这样就实现了对原图进行错位，分成两张图的操作，做差之后就是原图中每个像素点与相
        # 邻的下一个像素点的差。
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TVLoss_2(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss_2, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow(((x[:, :, 2:, :] - x[:, :, 1:h_x - 1, :]) - (x[:, :, 1:h_x - 1, :] - x[:, :, :h_x - 2, :])), 2).sum()
        w_tv = torch.pow(((x[:, :, :, 2:] - x[:, :, :, 1:w_x - 1]) - (x[:, :, :, 1:w_x - 1] - x[:, :, :, :w_x - 2])), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TVLoss_3(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss_3, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_hw = self._tensor_size(x[:, :, 1:h_x - 1, 1:w_x - 1])
        hw_tv = torch.abs(4 * x[:, :, 1:h_x - 1, 1:w_x - 1] - x[:, :, :h_x - 2, 1:w_x - 1] - x[:, :, 2:, 1:w_x - 1] -
                          x[:, :, 1:h_x - 1, :w_x - 2] - x[:, :, 1:h_x - 1, 2:]).sum()
        return self.TVLoss_weight * 2 * hw_tv / count_hw / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]