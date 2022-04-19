# 首先导入包
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import yaml
import anyconfig
from preprocess.aug.crop_paste import CutPaste,CutPasteNormal2
import collections
# This is for the progress bar.
from tqdm import tqdm
from torch.utils.data.sampler import WeightedRandomSampler
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

class GarbageData2(Dataset):
    def __init__(self, csv_path_train=None,csv_path_val=None,mode='train'):
        self.mode=mode
        if(mode=='train'):
            assert csv_path_train!=None
            self.data_info = pd.read_csv(csv_path_train)
        if (mode == 'valid'):
            assert csv_path_val != None
            self.data_info = pd.read_csv(csv_path_val)
        self.data_len =len(self.data_info)

        #打乱数据
        self.data_info = self.data_info.sample(frac=1).reset_index(drop=True)

        if mode == 'train':
            self.train_image =np.asarray(
            self.data_info.iloc[:, 1])  # self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len

            self.train_label = np.asarray(self.data_info.iloc[:, 2])
            self.train_big_label=np.asarray(self.data_info.iloc[:, 5])
            self.teacher_logit=(self.data_info.iloc[:,-1])
            self.train_datainfo = self.data_info.iloc[:, :]
            self.init_sampler()
            self.image_arr = self.train_image
            self.label_arr = self.train_label
            self.big_label_arr=self.train_big_label
        elif mode == 'valid':

            self.valid_image = np.asarray(self.data_info.iloc[:, 1])
            self.valid_label = np.asarray(self.data_info.iloc[:, 2])
            self.valid_big_label = np.asarray(self.data_info.iloc[:, 5])

            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
            self.big_label_arr=self.valid_big_label
        self.real_len = len(self.image_arr)
        print('Finished reading the {} set of garbage Dataset ({} samples found)'
              .format(mode, self.real_len))
    def init_sampler(self):
        count=collections.Counter(self.train_datainfo.iloc[:,2])
        result = list(collections.Counter(self.train_datainfo.iloc[:,2]).keys())
        # print(result)
        class_sample_count=np.ones(len(result))

        for type in result:
            class_sample_count[int(type)]=count[type]
        # print(class_sample_count)
        weight=1.0/class_sample_count
        target=np.array(self.train_datainfo.iloc[:,2], dtype=np.int32)
        assert len(target)==sum(class_sample_count)
        samples_weight = np.array([weight[t] for t in target])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        self.weight_sampler=sampler

    def __getitem__(self, index):
        open_path =self.image_arr[index]
        # 读取图像文件
        img_as_img = Image.open(open_path)
        if img_as_img.mode != 'RGB':
            img_as_img = img_as_img.convert('RGB')
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.mode == 'train':
            #训练数据增强

            # transform = transforms.Compose([
            #     transforms.Resize((224, 224)),
            #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            #     transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=mean, std=std)
            # ])
            transform =A.Compose(
                [
                    A.Resize(height=224, width=224),
                    A.RandomSizedCrop(min_max_height=(100, 224),height=224,width=224, p=1),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        # 得到图像的 string label
        label = self.label_arr[index]
        big_label=self.big_label_arr[index]
        # number label
        number_label = int(label)


        if(self.mode=="valid"):
            img_as_img = transform(img_as_img)
            return img_as_img, number_label,big_label,open_path  # 返回每一个index对应的图片数据和对应的label
        else:
            img_as_img=np.asarray(img_as_img)
            img_as_img = transform(image=img_as_img)
            logits=self.teacher_logit[index]
            logits_list=[]
            for logit in logits.split("_"):
                logits_list.append(eval(logit))

            return img_as_img, number_label, big_label, open_path,torch.tensor(logits_list)  # 返回每一个index对应的图片数据和对应的label
    def __len__(self):
        return self.real_len