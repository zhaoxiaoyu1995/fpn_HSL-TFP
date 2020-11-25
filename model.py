import torch
import torch.nn as nn
from fpn_head import FPNDecoder
from resnet import resnet18


class fpn(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18()
        self.head = FPNDecoder(encoder_channels=[512, 256, 128, 64])

    def forward(self, input):
        x = self.backbone(input)
        x = self.head(x)
        return x


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        print("Use CPU")
        device = torch.device('cpu')

    model = fpn().to(device)
    x = torch.zeros(8, 1, 640, 640).to(device)
    with torch.no_grad():
        y = model(x)
    print(y.size())
