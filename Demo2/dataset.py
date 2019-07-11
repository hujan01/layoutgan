""" 预处理数据集 """
import os 

import cv2
import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets
from matplotlib import pyplot as plt 

class MNISTLayoutDataset(data.Dataset):
    def __init__(self, path, element_num=128, gt_thresh=200):
        super(MNISTLayoutDataset, self).__init__()
        #self.train_data= torch.load(path + '/MNIST/processed/training.pt')[0]
        self.path = path + '/2/'
        self.fname = os.listdir(self.path)
        
        self.element_num = element_num
        self.gt_thresh = gt_thresh
        #self.w, self.h = self.train_data.shape[1], self.train_data.shape[2] #训练数据的高宽

    def __getitem__(self, idx):
        #img = self.train_data[idx]
        fimg = self.fname[idx]
        img = plt.imread(self.path + fimg)
        img = torch.from_numpy(img)
        self.w, self.h = img.shape
        img[img >= self.gt_thresh] = 255  #设置高阈值
        coord = ((img == 255).nonzero())  # 获取所有大于阈值的坐标 每张图片数目不一样
        ridx = torch.randint(low=0, high=coord.shape[0], size=(self.element_num,)) #会有重复的像素点
        coord = coord[ridx]
        coord = coord.type(torch.DoubleTensor)
        #对坐标进行归一化
        coord[:,0] /= self.w
        coord[:,1] /= self.h

        #类概率
        col_ones = torch.FloatTensor(self.element_num, 1).fill_(1)
        data = torch.cat((col_ones.float(), coord.float()), dim=1)
        return data

    def __len__(self):
        return len(self.fname)
        #return len(self.train_data)
        
