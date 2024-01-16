import os
import os.path
import numpy as np
import random
import cv2

from tqdm import tqdm
import torch
from torch.utils import data
from torch.utils.data import Dataset

# 解析txt文件，得到img和gt列表
def make_dataset(split=0, data_root=None, data_list=None, data_class=None, mode = "unset"):
    assert split in [0, 1, 2, 999]

    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    img_gt_class_list = []
    list_read = open(data_list, encoding='UTF-8-sig').readlines()
    print("\n split{}-{}-data-class-{}-successful......".format(split,mode,data_class))
    class_img_gt_dict = {}
    for sub_c in data_class:
        class_img_gt_dict[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split()
        image_name = os.path.join(data_root, line_split[0])
        temp = line_split[0].replace('Images/', 'GT/')
        gt_name = os.path.join(data_root, temp)
        gt_name = gt_name.replace('jpg', 'png').replace('bmp','png')
        image_class = line_split[1]
        item = (image_name, gt_name, image_class)
        img_gt_class_list.append(item)

        class_img_gt_dict[int(image_class)].append(item)

    print("Checking image&label pair {} list model! ".format(split))
    return img_gt_class_list, class_img_gt_dict



class SemData(Dataset):
    def __init__(self, split=0, shot=1, data_root=None, data_list=None, transform=None, mode='train'):
        assert mode in ['train', 'val', 'val']
        assert split in [0, 1, 2, 999]
        self.mode = mode
        self.split = split
        self.shot = shot
        self.data_root = data_root
        self.data_list = data_list
        # 根据split，选择训练和验证类别
        self.class_list = list(range(1, 13))
        if self.split == 2:
            self.train_class = list(range(1, 9))
            self.val_class = list(range(9, 13))
        elif self.split == 1:
            self.train_class = list(range(1, 5)) + list(range(9, 13))
            self.val_class = list(range(5, 9))
        elif self.split == 0:
            self.train_class = list(range(5, 13))
            self.val_class = list(range(1, 5))
        elif self.split == 999:
            self.train_class = list[1]
            self.val_class = list([1])

        if self.mode == 'train':    #得到img和gt列表
            self.img_gt_class_list, self.class_img_gt_dict = make_dataset(split, data_root, data_list, self.train_class, mode)
            assert len(self.class_img_gt_dict.keys()) == len(self.train_class)

        elif self.mode == 'val':
            self.img_gt_class_list, self.class_img_gt_dict = make_dataset(split, data_root, data_list, self.val_class, mode)
            assert len(self.class_img_gt_dict.keys()) == len(self.val_class)

        self.transform = transform

    def __len__(self):
        return len(self.img_gt_class_list)

    def __getitem__(self, index):
        # 读取query img/gt/class，貌似由于shuffle随机index
        query_img, query_gt, query_class = self.img_gt_class_list[index]

        query_rgb = cv2.imread(query_img, cv2.IMREAD_COLOR)
        query_rgb = cv2.cvtColor(query_rgb, cv2.COLOR_BGR2RGB)
        query_rgb = np.float32(query_rgb)

        query_mask = cv2.imread(query_gt, cv2.IMREAD_GRAYSCALE)
        query_mask = np.where(query_mask < 1, 0, 1)

        if query_rgb.shape[0] != query_mask.shape[0] or query_rgb.shape[1] != query_mask.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + query_img + " " + query_gt + "\n"))

        class_chosen = query_class

        # query img所属类的所有gt
        all_img_gt_list = self.class_img_gt_dict[int(class_chosen)]
        num_file = len(all_img_gt_list)

        # 随机读取support img/gt
        support_image_path_list = []
        support_gt_path_list = []
        support_idx_list = []

        for k in range(self.shot):
            support_idx = random.randint(1, num_file) - 1
            support_image_path = query_img
            support_label_path = query_gt
            while (support_image_path == query_img and support_label_path == query_gt) or support_idx in support_idx_list:
                support_idx = random.randint(1, num_file) - 1
                support_image_path, support_label_path, _ = all_img_gt_list[support_idx]

            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_gt_path_list.append(support_label_path)

        support_image_list = []
        support_label_list = []
        subcls_list = []
        # 读取support img/gt
        for k in range(self.shot):
            if self.mode == 'train':    #训练集实际类别索引位置，例如5，6，7，8，索引0，1，2，3，称为subcls
                subcls_list.append(self.train_class.index(int(class_chosen)))
            else:
                subcls_list.append(self.val_class.index(int(class_chosen)))

            support_image_path = support_image_path_list[k]
            support_label_path = support_gt_path_list[k]

            support_rgb = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
            support_rgb = cv2.cvtColor(support_rgb, cv2.COLOR_BGR2RGB)
            support_rgb = np.float32(support_rgb)

            support_mask = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            support_mask = np.where(support_mask < 1, 0, 1)


            if support_rgb.shape[0] != support_mask.shape[0] or support_rgb.shape[1] != support_mask.shape[1]:
                raise (RuntimeError(
                    "Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))
            support_image_list.append(support_rgb)
            support_label_list.append(support_mask)

        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot

        raw_label = query_mask.copy()
        
        
        # 变换
        if self.transform is not None:
            query_rgb, query_mask = self.transform(query_rgb, query_mask)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list[k],
                                                                              support_label_list[k])

        s_xs = support_image_list
        s_ys = support_label_list
        s_x = s_xs[0].unsqueeze(0)
        # ---- k-shot ---- ##拼接k个support[k,c,w,h]/[k,w,h]
        for i in range(1, self.shot):   
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)
        if self.mode == 'train':    #训练，数据集返回，查询img，gt，支持img，gt，索引类别，查询图像路径
            return query_rgb, query_mask, s_x, s_y, subcls_list, query_img
        else:                       #其他，数据集返回，查询img，gt，支持img，gt，索引类别，未变换的查询gt
            return query_rgb, query_mask, s_x, s_y, subcls_list, raw_label
