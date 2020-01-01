from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image


class Data(Dataset):
    def __init__(self, path):
        #传入图片路径
        self.path = path
        #创建数据集
        # self.dataset = []
        #将数据传入数据集
        self.dataset = os.listdir(self.path)
        # self.dataset.extend(os.path.join(self.path, x) for x in self.list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        #将数据集中的每一组数据分割
        strs = self.dataset[index]
        #找到图片路径
        img_path = os.path.join(self.path, strs)
        #将图片转为Tensor格式
        img_data = torch.Tensor(np.array(Image.open(img_path)) / 255. - 0.5)
        #将图片从H,W,C结构转换为C,H,W结构
        img_data = img_data.permute(2, 0, 1)


        return img_data

# if __name__ == '__main__':
#     path = r'E:\AI\GAN\faces'
#     data = Data(path)
#     print(data[1])