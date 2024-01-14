# import cv2
# >>>>>>>>>>pytorch Lab<<<<<<<<<
import torch
# import torch.nn.parallel
import torch.optim
import torch.utils.data
# >>>>>>>>>>define Lab<<<<<<<<<<
from util import dataset, transform

def dataset_train_val(args):
    # 均值标准差，尺度0-1改为0-255
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    assert args.split in [0, 1, 2, 999]  # split即当前选择的训练划分fold，12个类别分为3个fold
    train_loader = None
    val_loader = None
    
    if args.train:
        train_transform = [             #训练集变换：图像大小，旋转，高斯模糊，水平翻转，裁剪，totensor，标准化
            transform.RandScale([args.scale_min, args.scale_max]),
            transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, padding_label=args.padding_label),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, padding_label=args.padding_label),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)]

        train_transform = transform.Compose(train_transform)
        # 导入训练列表文件名，很快
        train_data = dataset.SemData(split=args.split,              # fold
                                        shot=args.shot,             # shot
                                        data_root=args.data_root,   # 数据目录
                                        data_list=args.train_list,  # 列表文件txt
                                        transform=train_transform,  # 变换
                                        mode='train')
        # 生成dataloader
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(train_data,                  #文件列表
                                                batch_size=args.batch_size,  #bs
                                                shuffle=True,                #打乱
                                                num_workers=args.workers,    #nworkers
                                                pin_memory=True,             #固定内存
                                                sampler=train_sampler,       #训练采样器为空
                                                drop_last=True)              #丢弃不完整batch
    # 如果评估，则生成评估val的dataloader
    if args.evaluate:
        if args.resized_val:    #评估时，改变val中图像大小，变换
            val_transform = transform.Compose([
                transform.Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
        else:
            val_transform = transform.Compose([
                transform.test_Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
        val_data = dataset.SemData(split=args.split,
                                      shot=args.shot,
                                      data_root=args.data_root,
                                      data_list=args.val_list,
                                      transform=val_transform,
                                      mode='val')
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=args.batch_size_val,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True,
                                                 sampler=val_sampler)        
    return train_loader, val_loader