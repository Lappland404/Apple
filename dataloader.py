import glob
import os
import random
from torch.utils.data import Dataset

import cv2
import torch


class UNet_Dataloader(Dataset):
    def __init__(self, datapath):
        self.datapath = datapath
        self.imgpath = glob.glob(os.path.join(datapath, 'image/*.png'))

    def __len__(self):
        return len(self.imgpath)

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgpath[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

if __name__ == "__main__":
    data = UNet_Dataloader(r"F:\code-new\learn\UNet\data\\train")
    print("数据个数：", len(data))
    train_loader = torch.utils.data.DataLoader(dataset=data,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)