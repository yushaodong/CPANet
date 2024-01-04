
import torch
from torch import nn
import torch.nn.functional as F

import model.resnet as models
import model.vgg as vgg_models
from model.CPP import CPP
from model.DP import DoneUp


def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4


class cpanet(nn.Module):
    def __init__(self, layers=50, classes=2, criterion=nn.CrossEntropyLoss(ignore_index=255),
                 pretrained=True, shot=1, vgg=False):
        super(cpanet, self).__init__()
        # assert layers in [50, 101, 152]
        # assert classes == 2
        from torch.nn import BatchNorm2d as BatchNorm

        self.criterion = criterion
        self.shot = shot
        self.vgg = vgg

        models.BatchNorm = BatchNorm

        if self.vgg:
            print('>>>>>>>>> Using VGG_16 bn <<<<<<<<<')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
            self.layer3, self.layer4 = get_vgg16_layer(vgg16)

        else:
            print('>>>>>>>>> Using ResNet {}<<<<<<<<<'.format(layers))
            if layers == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif layers == 101:
                resnet = models.resnet101(pretrained=pretrained)
            else:
                resnet = models.resnet152(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1,
                                        resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256

        if self.vgg:
            fea_dim = 512 + 256

        else:
            fea_dim = 1024 + 512


        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(reduce_dim, 2, kernel_size=3, padding=1, bias=False),
        )


        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )

        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )
        self.conv_2T1 = nn.Sequential(
            nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(reduce_dim , reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
        )
        self.conv_3T1 = nn.Sequential(
            nn.Conv2d(reduce_dim * 3, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
        )

        self.DP = DoneUp().cuda()

        self.REF = CPP(reduce_dim).cuda()

    def forward(self,
                x,
                s_x=torch.FloatTensor(4, 1, 3, 200, 200).cuda(),    #shot
                s_y=torch.FloatTensor(4, 1, 200, 200).cuda(),       #shot GT
                y=None):

        x_size = x.size()  # [4,3,200,200]
        h = int(x_size[-1])
        w = int(x_size[-2])


        with torch.no_grad():
            query_feat_0 = self.layer0(x)  # [4,128,50,50]
            query_feat_1 = self.layer1(query_feat_0)  # [4,256,50,50]
            query_feat_2 = self.layer2(query_feat_1)  # [4,512,25,25]
            query_feat_3 = self.layer3(query_feat_2)  # [4,1024,25,25]

            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2,
                                             size=(query_feat_3.size(2), query_feat_3.size(3)),
                                             mode='bilinear',
                                             align_corners=True)
        query_feat = torch.cat([query_feat_3, query_feat_2], dim=1)
        query_feat = self.down_query(query_feat)                        # 2，3特征结合后，1*1合并通道
        gt_list = []
        gt_down_list = []
        supp_feat_list = []
        supp_a_list = []
        # ----- 5shot Support----- #
        for i in range(self.shot):
            supp_gt = (s_y[:, i, :, :] == 1).float().unsqueeze(1)       #support GT
            gt_list.append(supp_gt)

            with torch.no_grad():                                       #support特征提取
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)

                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2,
                                                size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                mode='bilinear',
                                                align_corners=True)
            supp_gt_down = F.interpolate(supp_gt, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',align_corners=True)  #特征降采样
            gt_down_list.append(supp_gt_down)

            supp_feat = torch.cat([supp_feat_3, supp_feat_2], dim=1)
            supp_feat = self.down_supp(supp_feat)  # [4,256,25,25]      # 2，3特征结合后，1*1合并通道
            supp_feat_list.append(supp_feat)


            supp_feat_mask = supp_feat * supp_gt_down                   #mask*特征
            supp_a = self.REF(supp_feat_mask)  # [1,256,1,1]            #REF就是CPP，生成proxy

            supp_a_list.append(supp_a)

        supp_i = torch.zeros_like(supp_a_list[0])   # 【1，256，1，1】
        for i in range(self.shot):                  # 把k shots的proxy列表元素叠加
            supp_i += supp_a_list[i]
        supp_ap = supp_i/len(supp_i)                # 求均值
        supp_ap = supp_ap.expand(query_feat.shape[0], 256, query_feat.shape[-2], query_feat.shape[-1]) #proxy列展开为面

        query_supp = torch.cat([supp_ap, query_feat], dim=1)    #面合并FQ
        query_out = self.conv_2T1(query_supp)  # 1,256,200,200  #1*1,3*3，得到FSQ

        query_pred_mask = self.conv_3x3(query_out)
        query_pred_mask = F.interpolate(query_pred_mask, size=(h, w), mode='bilinear', align_corners=True)  # [1,256,200,200]
        query_pred_mask = self.DP(query_pred_mask)  #SSA模块

        query_pred_mask_save = torch.argmax(query_pred_mask[0].squeeze(0).permute(1, 2, 0), axis=-1).detach().cpu().numpy()
        query_pred_mask_save[query_pred_mask_save!=0] = 255
        query_pred_mask_save[query_pred_mask_save==0] = 0


        if self.training:                           #SA模块输出
            supp_pred_mask_list = []
            for i in range(self.shot):
                supp_s_i = supp_a_list[i]
                supp_feat_i = supp_feat_list[i]
                supp_s = supp_s_i.expand(supp_feat_i.shape[0], 256, supp_feat_i.shape[-2], supp_feat_i.shape[-1])
                supp_gcn = torch.cat([supp_feat_i, supp_s, supp_s] ,dim=1)
                supp_out = self.conv_3T1(supp_gcn)
                supp_out = F.interpolate(supp_out, size=(h, w), mode='bilinear', align_corners=True)  # [1,256,200,200]

                supp_pred_mask = self.cls(supp_out)
                supp_pred_mask_list.append(supp_pred_mask)


        alpah = 0.4
        if self.training:
            supp_loss_list = []
            loss = 0.
            for i in range(self.shot):      #支撑集loss
                supp_loss = self.criterion(supp_pred_mask_list[i], gt_list[i].squeeze(1).long())
                loss += supp_loss
            aux_loss = supp_loss/self.shot
            main_loss = self.criterion(query_pred_mask, y.long())

            return query_pred_mask.max(1)[1], main_loss + alpah * aux_loss
        else:
            return query_pred_mask, query_pred_mask_save


if __name__ == '__main__':

    model = cpanet()
    model = model.cuda()
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.4fM" % (total / 1e6))
    print(model)
    x = torch.FloatTensor(4, 3, 200, 200).cuda()
    print('input_size:', x.size())
    a, b, c = model(x)
    print(a.shape, b.shape, c.shape)

