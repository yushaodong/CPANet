import os
# import cv2
# >>>>>>>>>>pytorch Lab<<<<<<<<<
import torch
import torch.nn as nn
# import torch.nn.parallel
import torch.optim
import torch.utils.data
# >>>>>>>>>>define Lab<<<<<<<<<<
from model.CPANet import cpanet

def model_instance(args,logger,device):
    # 使用CE loss，计算损失时，忽略类别标签255
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = cpanet(layers=args.layers,                                  #层数
                   classes=2,                                           #类别数
                   criterion=nn.CrossEntropyLoss(ignore_index=255),     #忽略索引255的CE
                   pretrained=True,                                     #使用预训练模型
                   shot=args.shot,                                      #shot数
                   vgg=args.vgg)                                        #不使用vgg
    if args.layers == 50:
        # 冻结预训练模型参数
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
    # elif argss.layers == 'convtiny':
    #     model.convnet.parameters().requires_grad(False)

    # optimizer = torch.optim.SGD(      # 使用SGD，优化其余部分参数，设定lr/momentum/weight_decay
    optimizer = torch.optim.AdamW(
        [
            # {'params': model.parameters()},
            # {'params': model.convnet.parameters()},
            {'params': model.down_query.parameters()},
            {'params': model.down_supp.parameters()},
            {'params': model.REF.parameters()},
            {'params': model.cls.parameters()},
            {'params': model.conv_2T1.parameters()},
            {'params': model.conv_3x3.parameters()},
            {'params': model.DP.parameters()},
            {'params': model.conv_3T1.parameters()},
            # {'params': model.ybn1.parameters()},
            # {'params': model.ybn2.parameters()},
            # {'params': model.ybn3.parameters()},
        ],
        lr=args.base_lr, weight_decay=args.weight_decay)
        # lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    logger.info(model)
    logger.info("\033[1;36m >>>>>>Creating model ...\033[0m")
    logger.info("\033[1;36m >>>>>>Classes: {}\033[0m".format(args.classes))

    model = model.to(device)
    # 加载调优或test的权重文件
    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))
    # 加载恢复训练的文件
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())  #加载模型恢复到GPU
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    return model,optimizer,criterion
