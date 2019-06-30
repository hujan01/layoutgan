import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

from dataset import MNISTLayoutDataset
import model

def real_loss(D_out, device):
    #计算real图像损害
    batch_size = D_out.size(0)
    labels = torch.ones(batch_size).to(device)
    crit =nn.BCEWithLogitsLoss()
    assert (D_out.data.cpu().numpy().all() >= 0. and D_out.data.cpu().numpy().all() <= 1.)
    loss = crit(D_out.squeeze(), labels.squeeze())
    return loss

def fake_loss(D_out, device):
    #计算fake图像损失
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size).to(device) 
    crit = nn.BCEWithLogitsLoss()
    assert (D_out.data.cpu().numpy().all() >= 0. and D_out.data.cpu().numpy().all() <= 1.)
    loss = crit(D_out.squeeze(), labels.squeeze())
    return loss

def points_to_image(points):
    """ 绘制图像 """
    batch_size = points.size(0)
    images = []
    for b in range(batch_size):
        canvas = np.zeros((28, 28)) #生成背景图片
        image = points[b]  #第一张图片
        for point in image:
            if point[0] > 0.3: #看概率是否大于阈值
                x, y = int(point[1]*28), int(point[2]*28)
                x, y = min(x, 27), min(y, 27)
                canvas[x,y] = 255
        images.append(canvas)
    images = np.asarray(images)
    images_tensor = torch.from_numpy(images)
    return images_tensor

def main():
     # 设定参数
    element_num = 128
    cls_num = 1  
    geo_num = 2
    batch_size = 4
    lr = 0.0002
    num_epochs = 100

    #优化器参数
    beta1 = 0.99
    beta2 = 0.95

    # 选择运行环境
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # 加载数据集
    _ = datasets.MNIST(root='data', train=True, download=True, transform=None)
    train_data = MNISTLayoutDataset('data')
    train_loader = DataLoader(train_data, batch_size=batch_size)

    # 加载模型
    print("load model")
    gen = model.Generator(batch_size, element_num, geo_num, cls_num).to(device)
    dis = model.Discriminator(batch_size).to(device)

    # 定义优化器
    print("Initialize optimizers")
    g_optimizer = optim.Adam(gen.parameters(), lr, (beta1, beta2))
    d_optimizer = optim.Adam(dis.parameters(), lr/10, (beta1, beta2))

    # 设置为训练模式
    gen.train()
    dis.train()

    # 开始训练
    for epoch in range(1, num_epochs+1):
        for batch_idx, real_images in enumerate(train_loader, 1):

            real_images = real_images.to(device) 
            batch_size = real_images.size(0)
            print("[{}/{}]".format(batch_idx, epoch))
            # 训练判别器
            d_optimizer.zero_grad()
            print('start train discriminator')
            start = time.time()
            real_images = Variable(real_images)
            D_real = dis(real_images) #判断真实图像
            d_real_loss = real_loss(D_real, device) #计算真实图像损失

            # 随机初始化类别和位置信息
            z_cls = torch.FloatTensor(batch_size, element_num, geo_num).uniform_(0, 1) #均匀分布
            z_geo = torch.FloatTensor(batch_size, element_num, cls_num).normal_(0.5, 0.5) #正态分布
            z = torch.cat((z_cls, z_geo), 2).to(device)
            
            fake_images_d = gen(z) #生成fake图像
            D_fake = dis(fake_images_d) #判断fake图像
            d_fake_loss = fake_loss(D_fake, device) #计算fake图像损失

            # Total loss
            d_loss = d_real_loss + d_fake_loss
            #反向传播，迭代参数
            d_loss.backward()
            d_optimizer.step()
            print("batch discriminator train time {:.3f}".format(time.time()-start))
            # 训练生成器
            g_optimizer.zero_grad()
            print('start train generator')
            # 随机初始化
            z_cls = torch.FloatTensor(batch_size, element_num, cls_num).uniform_(0, 1)
            z_geo = torch.FloatTensor(batch_size, element_num, geo_num).normal_(0.5, 0.5)
            z = torch.cat((z_cls, z_geo), 2).to(device)

            fake_images_g = gen(z) #生成fake图像
            D_out = dis(fake_images_g) #判断fake图像

            g_loss = real_loss(D_out, device) 
            g_loss.backward()
            g_optimizer.step()
            print("batch train time {:.3f}".format(time.time()-start))
            #每迭代2次保存结果
            if batch_idx % 2 == 0: 
                #保存在文件夹中
                result_path = 'result'
                os.makedirs(result_path, exist_ok=True)
                test_samples = 9
                #随机初始化
                z_cls = torch.FloatTensor(test_samples, element_num, cls_num).uniform_(0, 1) #均匀分布
                z_geo = torch.FloatTensor(test_samples, element_num, geo_num).normal_(0.5, 0.5) #正态分布
                z = torch.cat((z_cls, z_geo), 2).to(device)

                generated_images = gen(z)
                #绘制生成图像
                generated_images = points_to_image(generated_images).view(-1, 1, 28, 28)
                save_image(generated_images, '{}/{}_{}.png'.format(result_path, epoch, batch_idx), nrow=3)
                print("[G: Loss = {:.4f}] [D: Total Loss = {:.4f} Real Loss = {:.4f} Fake Loss {:.4f}]".format(
                    g_loss, d_loss, d_real_loss, d_fake_loss))

if __name__ == '__main__':
    main()