import argparse
import os
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dataset import MNISTLayoutDataset
import model

def real_loss(D_out, device):
    #计算real图像损失
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
            if point[0] > 0: #看概率是否大于阈值
                x, y = int(point[1]*28), int(point[2]*28)
                x, y = min(x, 27), min(y, 27)
                canvas[x, y] = 255
        images.append(canvas)
    images = np.asarray(images)
    images_tensor = torch.from_numpy(images)
    return images_tensor

def show_lossHist(hist, path=None):
    """ 损失优化结果 """
    if not os.path.isdir(path):
        os.mkdir(path)
    fname = 'loss_hist.png'
    x = range(len(hist['D_losses'])) #x轴
    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.savefig(os.path.join(path, fname))
    
def main():
     # 设定参数
    element_num = 128
    cls_num = 1  
    geo_num = 2
    batch_size = 128
    lr = 0.0002
    num_epochs = 100

    #优化器参数
    beta1 = 0.5
    beta2 = 0.999
    
    #设置随机数种子
    manualSeed = random.randint(1, 10000) 
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # 选择运行环境
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # 加载数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    _ = datasets.MNIST(root='data', train=True, download=True, transform=transforms)
    train_data = MNISTLayoutDataset('data')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # 加载模型
    print("load model")
    gen = model.Generator(element_num, geo_num, cls_num).to(device)
    dis = model.RelationDiscriminator(batch_size, geo_num, cls_num, element_num).to(device)

    # 定义优化器
    print("Initialize optimizers")
    g_optimizer = optim.Adam(gen.parameters(), lr, (beta1, beta2))
    #g_optimizer = optim.SGD(gen.parameters(), lr)
    d_optimizer = optim.Adam(dis.parameters(), lr/10, (beta1, beta2))
    #d_optimizer = optim.SGD(dis.parameters(), lr/10)

    # 设置为训练模式
    gen.train()
    dis.train()

    loss_hist = {}
    loss_hist['D_losses'] = []
    loss_hist['G_losses'] = []
    loss_hist['per_epoch_times'] = []
    loss_hist['total_times'] = []

    z_cls = torch.FloatTensor(torch.ones(batch_size, element_num, cls_num))#类别都是1
    z_geo = torch.FloatTensor(batch_size, element_num, geo_num).normal_(0.5, 0.15) #正态分布
    fixed_z = torch.cat((z_cls, z_geo), 2).to(device)

    # 开始训练
    print('*****************************')
    print('start train!!!')
    start_time = time.time()
    for epoch in range(num_epochs):
        #scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=2, gamma=0.1)
        D_losses, G_losses = [], []
        epoch_start_time = time.time()
        for batch_idx, real_images in enumerate(train_loader, 1):
            #输出真实图像，观察提取像素点效果,这里只显示第一个批次
            if batch_idx == 1:
                imgs = real_images[:16, :, :]
                real_imgs = points_to_image(imgs).view(-1, 1, 28, 28)
                save_image(real_imgs, 'real_img.png', nrow=4)

            real_images = real_images.to(device) 
            batch_size = real_images.size(0)
            # 训练判别器
            d_optimizer.zero_grad()

            real_images = Variable(real_images)
            D_real = dis(real_images) #判断真实图像
            d_real_loss = real_loss(D_real, device) #计算真实图像损失

            # 随机初始化类别和位置信息
            z_cls = torch.FloatTensor(batch_size, element_num, cls_num).uniform_(0, 1) #均匀分布
            z_geo = torch.FloatTensor(batch_size, element_num, geo_num).normal_(0.5, 0.15) #正态分布
            z = torch.cat((z_cls, z_geo), 2).to(device)
            
            fake_images_d = gen(z) #生成fake图像
            D_fake = dis(fake_images_d) #判断fake图像
            d_fake_loss = fake_loss(D_fake, device) #计算fake图像损失

            # Total loss
            d_loss = d_real_loss + d_fake_loss

            #反向传播，迭代参数
            d_loss.backward()
            d_optimizer.step()
            D_losses.append(d_loss.item()) #一个epoch中的损失


            g_optimizer.zero_grad()
            # 随机初始化
            z_cls = torch.FloatTensor(batch_size, element_num, cls_num).uniform_(0, 1)
            z_geo = torch.FloatTensor(batch_size, element_num, geo_num).normal_(0.5, 0.15)
            z = torch.cat((z_cls, z_geo), 2).to(device)

            fake_images_g = gen(z) #生成fake图像
            D_out = dis(fake_images_g) #判断fake图像

            g_loss = real_loss(D_out, device) 
            g_loss.backward()
            g_optimizer.step()
            G_losses.append(g_loss.item())#一个epoch中的损失

        epoch_end_time = time.time()
        per_epoch_time = epoch_end_time - epoch_start_time
        print("[{}/{}] --time: {:.2f}, d_loss: {:.6f}, g_loss: {:.6f}".format(epoch+1, \
                                                                            num_epochs, per_epoch_time, d_loss, g_loss))
        #测试部分
        result_path = 'result_image'
        if not os.path.isdir(result_path):
            os.mkdir(result_path)

        generated_images = gen(fixed_z)
        generated_images = points_to_image(generated_images[:16, :, :]).view(-1, 1, 28, 28)
        save_image(generated_images, '{}/{}.png'.format(result_path, epoch+1, ), nrow=4)

        loss_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        loss_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        loss_hist['per_epoch_times'].append(per_epoch_time)
    
    end_time = time.time()
    total_time = end_time - start_time
    loss_hist['total_times'].append(total_time)

    print('avg per epoch time: {:.2f}, total {} epochs time: {:.2f}'.format(
                                                                    torch.mean(torch.FloatTensor(loss_hist['per_epoch_times'])),\
                                                                    num_epochs, total_time))
    show_lossHist(loss_hist, path='train_loss_image')
    print("finish train!!!")
    print("*******************************")

if __name__ == '__main__':
    main()