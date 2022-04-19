import random
import math
from torchvision import transforms
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
class CutPaste(object):
    """Base class for both cutpaste variants with common operations"""

    def __init__(self, colorJitter=0.1, transform=None):
        self.transform = transform

        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(brightness=colorJitter,
                                                      contrast=colorJitter,
                                                      saturation=colorJitter,
                                                      hue=colorJitter)

    def __call__(self, img):
        # apply transforms to both images
        if self.transform:
            img = self.transform(img)

        return  img

class CutPasteNormal(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """

    def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, **kwags):
        super(CutPasteNormal, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img,bg_img):
        # TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img.size[0]
        w = img.size[1]
        resize=transforms.Resize(50)
        resizeimg=resize(img)
        deco=resizeimg.copy()
        deco=np.asarray(deco)
        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h

        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1 / self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        cut_w = int(round(math.sqrt(ratio_area * aspect))*0.9)
        cut_h = int(round(math.sqrt(ratio_area / aspect))*0.9)

        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int((random.uniform(0, (h - cut_h))))
        from_location_w = int((random.uniform(0, (w - cut_w))))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)

        if self.colorJitter:
            patch = self.colorJitter(patch)

        to_location_h = int((random.uniform(0, (h - cut_h))))
        to_location_w = int((random.uniform(0, (w - cut_w))))

        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        insert_box2 = [to_location_w, to_location_h, to_location_w + deco.shape[0], to_location_h + deco.shape[1]]
        augmented = img.copy()
        # bg_img.paste(patch, insert_box)
        print(resizeimg)
        bg_img.paste(resizeimg, insert_box2)

        return super().__call__(img, bg_img)


class CutPasteNormal2(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """

    def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3,bg_transform=None,gab_transform=None, **kwags):
        super(CutPasteNormal2, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.bg_transform=bg_transform
        self.gab_transform=gab_transform

    def __call__(self, img,bg_img):
        # TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        img=self.gab_transform(img)
        bg_img=self.bg_transform(bg_img)
        h = img.size[0]
        w = img.size[1]
        bgh = bg_img.size[0]
        bgw = bg_img.size[1]
        new_h=int(bgh/14)
        radio=h/w
        new_w=int(new_h/radio)
        print(new_h)
        print(new_w)

        resize=transforms.Resize((new_h,new_w))
        resizeimg=resize(img)
        # deco=resizeimg.copy()
        # deco=np.asarray(deco)
        # decobg=np.asarray(bg_img)

        to_location_h = int(random.uniform(bg_img.size[1] / 2 -80 , bg_img.size[1] / 2+80 ))
        to_location_w = int(random.uniform(bg_img.size[0] / 2 -80, bg_img.size[0] / 2 +80))
        # print(to_location_h," ",to_location_w)

        insert_box2 = [to_location_w, to_location_h, to_location_w +resizeimg.size[0], to_location_h + resizeimg.size[1]]
        print(to_location_w)
        print(to_location_h)

        # augmented = img.copy()
        # bg_img.paste(patch, insert_box)
        # print(resizeimg)
        bg_img.paste(resizeimg, insert_box2)

        return super().__call__( bg_img)
if __name__ == '__main__':
    img_path="D:/dataset_garb/garbage_classify_v2/train_data_v2/img_3902.jpg"
    bg_path="D:/dataset_garb/collect-images/collect-images/8752-post.jpg"
    img_as_img = Image.open(img_path)
    img_as_bg=Image.open(bg_path)
    plt.imshow(img_as_img)
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
        # transforms.ToTensor()
    ])
    bg_transform=transforms.Compose([
        transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
        # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
        # transforms.ToTensor()
    ])
    gab_transform=transforms.Compose([
        transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
        transforms.RandomHorizontalFlip(p=0.2),  # 随机水平翻转 选择一个概率

        # transforms.ToTensor()
    ])

    mycp=CutPasteNormal2(transform=transform,bg_transform=bg_transform,gab_transform=gab_transform)
    img=mycp(img_as_img,img_as_bg)
    plt.imshow(img)
    plt.show()
