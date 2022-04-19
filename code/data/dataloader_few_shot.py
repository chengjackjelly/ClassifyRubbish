# 首先导入包
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from torchvision import transforms
from PIL import Image
import os
import yaml
import anyconfig
"""
实现功能：4 ways 1 shot
dataset getitem 返回
support set:[img1,img2,img3,img4] 分别来自四种类别
            [y_1（1000）,y_2（0100）,y_3(0010),y_4(0001)]
target set :[img_t]  第二种类型垃圾
            [y_t(0100)] 
"""
class GarbageDataFewShot(Dataset):
    def __init__(self, csv_path, file_path,root_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256,episodes=100):
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
        self.data_info = pd.read_csv(csv_path,header=None)  # header=None是去掉表头部分
        self.data_info=shuffle(self.data_info).reset_index(drop=True)

        # 计算 length

        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))
        # self.shot_df=[]
        self.train_support_arr_img=[]
        self.train_support_y=[]

        self.support_arr_img=[] ###img path per shot
        self.support_y=[]   ###label per shot
        if mode == 'train':
            # 第一列包含图像文件的名称

            self.train_image = np.asarray(
                self.data_info.iloc[1:self.train_len, 1])  # self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            # 第二列是图像的 label
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len, -1])
            self.image_arr = self.train_image
            self.label_arr = self.train_label


            self.df=self.data_info.iloc[1:self.train_len]
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, -1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image

        self.classes_dict=["1","2","3","4"]
        # print('Finished reading the {} set of garbage Dataset ({} samples found)'
        #       .format(mode, self.real_len))
        self.episodes=episodes
        self.create_episodes(self.episodes)

    def create_episodes(self,episodes):
        self.support_set_x_batch = []
        self.target_x_batch = []
        for _ in np.arange(episodes):
            support_set_x = []
            target_x = []
            a = [0, 1, 2, 3]
            np.random.shuffle(a)
            a.append(np.random.choice(a))
            for i in range(5): ###4way+1shot

                tmpdf=self.df[self.df.iloc[:,-1]==str(a[i])]
                tmpdf = tmpdf.reset_index(drop=True)

                choice=np.random.choice(tmpdf.index)

                img_path= tmpdf.iloc[choice][1]
                label=tmpdf.iloc[choice][5]

                assert label==str(a[i])
                support_set_x.append(img_path)
                target_x.append(label)
            self.support_set_x_batch.append(support_set_x)
            self.target_x_batch.append(target_x)

    def __getitem__(self, index):
        support_set_x = torch.FloatTensor(4, 3, 224, 224)
        target_x =torch.FloatTensor(1,3,224,224)
        support_set_y=np.zeros((4), dtype=np.int)
        target_y = np.zeros((1), dtype=np.int)
        for i in range(5):

            open_path=self.root_path + self.support_set_x_batch[index][i][3:]
            img_as_img = Image.open(open_path)
            if img_as_img.mode != 'RGB':
                print("not rgb", open_path)
                img_as_img = img_as_img.convert('RGB')
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

                if(i!=4):
                    #
                    support_set_x[i]=img_as_img
                    support_set_y[i]=int(self.target_x_batch[index][i])
                else:
                    target_x[0]=img_as_img
                    target_y[0]=int(self.target_x_batch[index][i])
                    print(target_y[0])





        return support_set_x,torch.IntTensor(support_set_y),target_x, torch.IntTensor(target_y)

    def __len__(self):

        return self.episodes
        pass
if __name__ == '__main__':
    """
        test
        """
    root = "D:/dataset_garb/"  # 根目录

    cfg_path = os.path.join(root + "code/config/cfg.yaml")
    assert (os.path.exists(cfg_path))
    config = anyconfig.load(open(cfg_path, 'rb'))
    train_csv_path = config['train_csv_path']
    train_img_path = config['train_img_path']

    train_path = os.path.join(
        root + train_csv_path
    )
    img_path = os.path.join(
        root + train_img_path
    )
    assert (os.path.exists(train_path))
    assert (os.path.exists(img_path))

    train_dataset = GarbageDataFewShot(train_path,root_path="D:/dataset_garb/", file_path=img_path, mode='train')
    # val_dataset = GarbageDataFewShot(train_path, img_path, mode='valid')
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    # val_loader = torch.utils.data.DataLoader(
    #     dataset=val_dataset,
    #     batch_size=8,
    #     shuffle=True,
    #     num_workers=0
    # )
    dataiter = iter(train_loader)
    test=dataiter.next()
    print(test)
    from model.network import  *

    dcn = DCN(num_class=4, num_support=4, num_query=1, num_embedding_class=10, with_variation=True)
    x = dcn(test[0],test[2])



from torchtext.legacy.data import Field