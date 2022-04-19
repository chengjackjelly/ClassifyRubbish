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
def get_train_transform(img,bgimg):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        #
        # # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
        # transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
    ])
    bg_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
        # transforms.ToTensor()
    ])
    gab_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(p=0.2),  # 随机水平翻转 选择一个概率
        # transforms.ToTensor()
    ])

    mycp = CutPasteNormal2(transform=transform, bg_transform=bg_transform, gab_transform=gab_transform)
    img = mycp(img, bgimg)
    return img
def weight_sampler():
    pass

def show_img(img):
    plt.imshow(img)
    plt.show()
def get_val_transform(img,bgimg):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    bg_transform = transforms.Compose([
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(p=0.2),  # 随机水平翻转 选择一个概率
        # transforms.ToTensor()
    ])
    gab_transform = transforms.Compose([
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(p=0.2),  # 随机水平翻转 选择一个概率
        # transforms.ToTensor()
    ])

    mycp = CutPasteNormal2(transform=transform, bg_transform=bg_transform, gab_transform=gab_transform)
    img = mycp(img, bgimg)

    return img
class GarbageData(Dataset):
    def __init__(self, csv_path, file_path,root_path,bg_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        """
        csv_path:储存路径信息
        file_path:图片所在路径


        """
        self.root_path=root_path
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.mode = mode
        # 读取 csv 文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path)  # header=None是去掉表头部分
        # self.data_info =self.data_info.iloc[0:10,:]
        self.data_info=self.data_info.sample(frac=1).reset_index(drop=True)
        # 计算 length

        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == 'train':
            # 第一列包含图像文件的名称
            self.train_image = np.asarray(
                self.data_info.iloc[0:self.train_len, 1])  # self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            # 第二列是图像的 label
            self.train_label = np.asarray(self.data_info.iloc[0:self.train_len, -1])
            self.train_datainfo=self.data_info.iloc[0:self.train_len]
            self.init_sampler()
            self.image_arr = self.train_image
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, -1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image

        self.real_len = len(self.image_arr)
        print('Finished reading the {} set of garbage Dataset ({} samples found)'
              .format(mode, self.real_len))


        #加载背景图
        self.bg_root=bg_path
        self.bg_img=os.listdir(bg_path)
        self.bg_len=len(self.bg_img)
    def init_sampler(self):
        count=collections.Counter(self.train_datainfo.iloc[:,-1])

        result = list(collections.Counter(self.train_datainfo.iloc[:,-1]).keys())
        # print(result)
        class_sample_count=np.ones(len(result))

        for type in result:
            class_sample_count[int(type)]=count[type]
        # print(class_sample_count)
        weight=1.0/class_sample_count
        target=np.array(self.train_datainfo.iloc[:,-1], dtype=np.int32)
        assert len(target)==sum(class_sample_count)

        samples_weight = np.array([weight[t] for t in target])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        self.weight_sampler=sampler
    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        open_path=self.root_path+self.image_arr[index][3:]

        # 读取图像文件
        img_as_img = Image.open(open_path)
        if img_as_img.mode !='RGB':
            print("not rgb",open_path)
            img_as_img = img_as_img.convert('RGB')
        # 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        #         if img_as_img.mode != 'L':
        #             img_as_img = img_as_img.convert('L')

        # 设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        if self.mode == 'train':

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
                transforms.ToTensor()
            ])
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

        img_as_img = transform(img_as_img)

        if self.mode == 'test':
            return img_as_img
        else:
            # 得到图像的 string label
            label = self.label_arr[index]
            # number label
            number_label = int(label)

            return img_as_img, number_label  # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.real_len





if __name__ == '__main__':
    """
    test
    """
    root="D:/dataset_garb/" #根目录

    cfg_path=os.path.join(root+"code/config/cfg.yaml")
    assert (os.path.exists(cfg_path))
    config = anyconfig.load(open(cfg_path, 'rb'))
    train_csv_path=config['train_csv_path']
    train_img_path=config['train_img_path']
    bg_root_path=config['bg_root_path']


    train_path=os.path.join(
        root+train_csv_path
    )
    img_path=os.path.join(
        root+train_img_path
    )
    assert (os.path.exists(train_path))
    assert (os.path.exists(img_path))
    train_dataset = GarbageData(train_path, img_path,root,bg_root_path, mode='train')
    val_dataset = GarbageData(train_path, img_path,root,bg_root_path, mode='valid')
