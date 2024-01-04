import torch.nn as nn
import torch.nn.functional as F
import math
import torch

from model.CBAM import Cbam


class DoneUp(nn.Module):
    def __init__(self):
        super(DoneUp, self).__init__()
        reduce_channels = 256
        self.Done1 = nn.Sequential(
            nn.Conv2d(in_channels=reduce_channels, out_channels=reduce_channels, kernel_size=(2, 2), stride=2, padding=0,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

        self.Done2 = nn.Sequential(
            nn.Conv2d(in_channels=reduce_channels, out_channels=reduce_channels, kernel_size=(2, 2), stride=2, padding=0,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(reduce_channels * 3, reduce_channels, kernel_size=(1, 1),  padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(reduce_channels, reduce_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),

        )

        self.res_conv = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=(3,3),padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),

        )
        self.cls = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=(1, 1))
        )

        self.Cbam = Cbam(256)

        self._init_weight()


    def forward(self, x):
        x1 = self.Done1(x)
        x1_up = F.interpolate(x1, scale_factor=2, mode='bilinear',align_corners=True)
        x2 = self.Done2(x1)
        x2_up = F.interpolate(x2, scale_factor=4, mode='bilinear',align_corners=True)
        x_cat = torch.cat([x, x1_up, x2_up], dim=1)
        x_x = self.conv_cat(x_cat)
        x_x_r = self.res_conv(x_x)  # [4,256,200,200]
        x_atten = self.Cbam(x_x)    # [4,256,200,200]
        out = x_x_r + x_atten       # [4,256,200,200]
        out = self.cls(out)         # [4,2,200,200]

        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':

    model = DoneUp()
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))
    input = torch.randn(4, 256, 200, 200)
    print('input_size:', input.size())
    out = model(input)
    print('out',out.shape)



