
import torch
from torch import nn
from torch.nn import functional as F


class CPP(nn.Module):
    def __init__(self, in_channels, sub_sample=True, bn_layer=True):
        super(CPP, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = in_channels // 2

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.peak_pool = nn.AdaptiveMaxPool2d(1)
        self.cos_similarity = nn.CosineSimilarity()
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=(1, 1), stride=(1, 1), padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=(1, 1), stride=(1, 1), padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=(1, 1), stride=(1, 1), padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g,
                                   nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi,
                                     nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        x3 = self.avg_pool(z)

        return x3




if __name__ == '__main__':
    import torch

    img = torch.ones(4, 256, 25, 25)
    net = CPP(256)
    out1 = net(img)
    print(out1.size())
    total = sum(p.numel() for p in net.parameters())
    print("Total params: %.4fM" % (total / 1e6))
