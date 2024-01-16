import os
import random
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from val import validate
from util.dataset_train_val import dataset_train_val
from tensorboardX import SummaryWriter
# from model.hotmap import MSRGNet
from model.CPANet import cpanet
from util import config,util

# GPU_ID = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

device = torch.device('cuda:0')

ymlpath='config/SSD/test-defect.yaml'
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    # parser.add_argument('--config', type=str, default='config/SSD/test-defect.yaml', help='config file')
    parser.add_argument('--config', type=str, default=ymlpath, help='config file')
    args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg

def main_worker(argss):
    global args
    args = argss

    import time
    current_time = time.localtime()
    time_string = time.strftime("%Y-%m-%d__%H-%M-%S", current_time)
    resultfolder = os.path.splitext(os.path.basename(args.val_list))[0]
    resultfolder = 'exp/result_'+resultfolder + time_string
    util.check_makedirs(resultfolder)
    args.resultfolder = resultfolder
                
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = cpanet(layers=args.layers,
                    classes=2,
                    criterion=nn.CrossEntropyLoss(ignore_index=255),
                    pretrained=True,
                    shot=args.shot,
                    vgg=args.vgg)

    global logger, writer
    logger = config.get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    print(args)

    model = model.to(device)

    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    _,val_loader = dataset_train_val(args)

    loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou = validate(val_loader, model, criterion,logger, args)

if __name__ == '__main__':
    args = get_parser()
    assert args.classes > 1
    if args.manual_seed is not None:
        # cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)
    main_worker(args)
