import sys
import torch
import torch.nn as nn
import os
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import anyconfig
sys.path.append("..")
from data.datalorder import GarbageData
from data.datalorderS import GarbageData2
from data.datalorder import get_val_transform
from data.datalorder import show_img
from model.main import *
def predict():
    os.chdir("..")
    print(os.getcwd())
    mmodel=get_res_model_my(2,model_path="weight/gabtest.pth")
    print(mmodel)
    bg_path = "D:/dataset_garb/新建文件夹/8784-post.jpg" #
    # bg_path = "D:/dataset_garb/新建文件夹/8784-post.jpg"  #无明显可回收
    # bg_path = "D:/dataset_garb/新建文件夹/8765-post.jpg" #有明显可回收
    # bg_path = "D:/dataset_garb/新建文件夹/9593-pre.jpg" #无明显可回收 曝光
    # bg_path = "D:/dataset_garb/新建文件夹/8761-post.jpg"
    # bg_path = "D:/dataset_garb/新建文件夹/8783-post.jpg"
    img_path="D:/dataset_garb/garbage_classify_v2/train_data_v2/img_4469.jpg"

    img_as_img = Image.open(img_path)
    img_as_bg=Image.open(bg_path)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    # img_as_input=get_val_transform(img=img_as_img,bgimg=img_as_bg)
    # img_as_input=torch.unsqueeze(img_as_input,0)
    pure_input=torch.unsqueeze(transform(img_as_bg),0)
    _,_,_,_,result=mmodel(pure_input)
    show_img(img_as_bg)
    print(result)
    result=torch.softmax(result,-1)
    print(result)
    pass
if __name__ == '__main__':
    predict()