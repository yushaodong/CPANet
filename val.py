# >>>>>>>>>>python Lab<<<<<<<<<
import random
import time
import numpy as np
# >>>>>>>>>>pytorch Lab<<<<<<<<<
import torch
import torch.nn.functional as F
# import torch.nn.parallel
import torch.optim
import torch.utils.data
# >>>>>>>>>>define Lab<<<<<<<<<<
from util.util import AverageMeter, intersectionAndUnionGPU

def validate(val_loader, model, criterion, logger, args):
    logger.info('\n\033[1;36m>>>>>>>>>>>>>>>> Start Evaluation <<<<<<<<<<<<<<<<<<\033[0m]')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    split_gap = 4
    class_intersection_meter = [0] * split_gap
    class_union_meter = [0] * split_gap

    if args.manual_seed is not None and args.fix_random_seed_val:
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    model.eval()
    end = time.time()
    test_num = len(val_loader)
    assert test_num % args.batch_size_val == 0
    iter_num = 0
    for e in range(1): #10次重复评估，val-loader
        for i, (input, target, s_input, s_mask, subcls, ori_label) in enumerate(val_loader):
            if (iter_num - 1) * args.batch_size_val >= test_num:
                break #循环次数>loader长度，莫名其妙，浪费时间
            iter_num += 1
            data_time.update(time.time() - end)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True)
            
            ori_label = ori_label.cuda(non_blocking=True)
            start_time = time.time()
            #给定支持/查询样本的img/GT
            output, pred = model(s_x=s_input, s_y=s_mask, x=input, y=target)
            model_time.update(time.time() - start_time)
            
            if args.train==False and args.evaluate==True:
                save_img_gt_pred(input,target,pred, args.resultfolder,i)

            if args.ori_resize:
                longerside = max(ori_label.size(1), ori_label.size(2))
                backmask = torch.ones(ori_label.size(0), longerside, longerside).cuda() * 255
                backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                target = backmask.clone().long()    #由原始标签得到的target
            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
            
            loss = criterion(output, target)    #val loss
            loss = torch.mean(loss)

            output = output.max(1)[1]           #每一行最大值所在位置

            # 交 并 target （面积）
            intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)  # output索引位置，target，2类，忽略255
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)

            subcls = subcls[0].cpu().numpy()[0]
            class_intersection_meter[(subcls - 1) % split_gap] += intersection[1]
            class_union_meter[(subcls - 1) % split_gap] += union[1]

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % args.print_freq == 0):
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num * args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    class_iou_class = []
    class_miou = 0

    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou * 1.0 / len(class_intersection_meter)
    logger.info('mIoU---Val result: mIoU {:.4f}.'.format(class_miou))

    # for i in range(split_gap):
    #     logger.info('Class_{} Result: iou {:.4f}.'.format(i + 1, class_iou_class[i]))
    logger.info('Class_{} Result: iou {}.'.format(split_gap, class_iou_class))

    logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    logger.info('argClasses_{} Result: iou/accuracy {}/{}.'.format(args.classes, iou_class, accuracy_class))
    logger.info('\n\033[1;36m<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<\033[0m\n')

    print('avg inference time: {:.4f}, count: {}'.format(model_time.avg, test_num))

    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou


import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
def save_img_gt_pred(input,target,pred,resultfolder,i):
    input_array = input.cpu().squeeze().permute(1, 2, 0).numpy()
    input_image = ((input_array + 1) * 127.5 ).astype('uint8') 
    # cv2.imwrite("C:/1.bmp",input_image)
    array = target.cpu().squeeze().numpy() 
    image = (array*255).astype('uint8')
    # cv2.imwrite("C:/2.bmp",image)
    save_path = r'./{}/{}.png'.format(resultfolder,i+1)
    # cv2.imwrite("C:/3.png",pred)
    h, w, _ = input_image.shape
    result=np.zeros((h,w*3,3),dtype=np.uint8)
    result[:, 0:w] = input_image
    result[:, w:2*w] = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    result[:, 2*w:3*w] = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(save_path,result)    