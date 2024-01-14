# >>>>>>>>>>python Lab<<<<<<<<<
import argparse
import os
import random
import time
import numpy as np
from tqdm import tqdm
# >>>>>>>>>>pytorch Lab<<<<<<<<<
import torch
import torch.backends.cudnn as cudnn
# import torch.nn.parallel
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
# >>>>>>>>>>define Lab<<<<<<<<<<
from model import model_instance
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU
from util import config
from util.dataset_train_val import dataset_train_val
from val import validate

# import cv2
# cv2.ocl.setUseOpenCL(False)
# cv2.setNumThreads(0)

# GPU_ID = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
device = torch.device('cuda:0')

# def worker_init_fn(worker_id):
#     random.seed(args.manual_seed + worker_id)

def main_process():
    return True

def main_worker(argss):
    global args, logger, writer
    args = argss
    print(args)
    
    logger = config.get_logger()
    current_time = time.localtime()
    time_string = time.strftime("%Y-%m-%d__%H-%M", current_time)
    writer = SummaryWriter(f'{args.save_path}{time_string}')
    text = ''.join([f'{key}: {value}\n\r' for key, value in argss.items()]) #参数写入tensorboard
    writer.add_text('Parameters', text, 0)  

    model,optimizer,criterion = model_instance(args,logger,device)
    train_loader,val_loader = dataset_train_val(args)

    # 评价指标
    max_iou = 0.
    max_fbiou = 0
    best_epoch = 0
    filename = 'CPANet.pth'
    # 遍历epochs
    for epoch in range(args.start_epoch, args.epochs):
        if args.fix_random_seed_val:    #固定随机种子
            torch.cuda.manual_seed(args.manual_seed + epoch)
            np.random.seed(args.manual_seed + epoch)
            torch.manual_seed(args.manual_seed + epoch)
            torch.cuda.manual_seed_all(args.manual_seed + epoch)
            random.seed(args.manual_seed + epoch)
        # 训练1个epoch
        epoch_log = epoch + 1
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, epoch)
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        if args.evaluate:
            loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou = validate(val_loader, model, criterion, logger, args)
            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                writer.add_scalar('class_miou_val', class_miou, epoch_log)
            if class_miou > max_iou:
                max_iou = class_miou
                best_epoch = epoch
                if os.path.exists(filename):
                    os.remove(filename)
                filename = args.save_path + '/train_epoch_' + str(epoch) + '_' + str(max_iou) + '.pth'
                logger.info('Saving checkpoint to: ' + filename)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

            max_fbiou = mIoU_val if mIoU_val > max_fbiou else max_fbiou # max_iou和max_fbiou似乎不是一个episode
            logger.info('Best Epoch {:.1f}, Best IOU {:.4f} Best FB-IoU {:4F}'.format(best_epoch, max_iou, max_fbiou))

    filename = args.save_path + '/final.pth'
    logger.info('Saving checkpoint to: ' + filename)
    torch.save({'epoch': args.epochs, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

def train(train_loader, model, optimizer, epoch):
    if main_process():
        logger.info("\n\033[1;36m >>>>>>>>>>>>>>>> Start Train <<<<<<<<<<<<<<<<<< \033[0m")

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # main_loss_meter = AverageMeter()
    # aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (input, target, s_input, s_mask, subcls, rawlable) in tqdm(enumerate(train_loader)):
        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1

        # poly 策略调整学习率
        if args.base_lr > 1e-6:
            poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power,
                               warmup=args.warmup, warmup_step=len(train_loader) // 2)

        s_input = s_input.to(device,non_blocking=True)
        s_mask = s_mask.to(device,non_blocking=True)
        input = input.to(device,non_blocking=True)
        target = target.to(device,non_blocking=True)

        output, main_loss = model(s_x=s_input, s_y=s_mask, x=input, y=target)
        loss = main_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #交 并 target （面积）
        intersection, union, targetarea = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, targetarea = intersection.cpu().numpy(), union.cpu().numpy(), targetarea.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(targetarea)
        #分割精度，分割正确结果/GT
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        # main_loss_meter.update(main_loss.item(), args.batch_size)
        loss_meter.update(loss.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        # 预估时间
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
    
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch:[{}/{}] Loader:[{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        # 'MainLoss {main_loss_meter.val:.4f} '
                        # 'AuxLoss {aux_loss_meter.val:.4f} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                        #   main_loss_meter=main_loss_meter,
                                                        #   aux_loss_meter=aux_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)        #mIoU
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (targetarea + 1e-10)), current_iter)   #mAcc
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)                                     #

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
            epoch, args.epochs, mIoU, mAcc, allAcc))
        logger.info('argClasses_{} Result: iou： {} - accuracy： {}.'.format(
            args.classes, iou_class, accuracy_class)) #前景/背景
        logger.info('\n\033[1;36m<<<<<<<<<<<<<<<<< End Train <<<<<<<<<<<<<<<<<\033[0m\n')
    return loss_meter.avg, mIoU, mAcc, allAcc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Few-Shot Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/SSD/fold1_train.yaml', help='config file')
    args = parser.parse_args()
    args = config.load_cfg_from_cfg_file(args.config)
    
    assert args.classes > 1
    # assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0

    if args.manual_seed is not None:    #设定随机种子
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)
    main_worker(args)