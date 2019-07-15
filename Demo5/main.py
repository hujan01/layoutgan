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
from torchvision import datasets , transforms
from torchvision import utils as vutils
from torch import autograd
from dataset import MNISTLayoutDataset
import model

def points_to_image(points):
    """ 坐标点转成像素图 """
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

def normal_init(m, mean, std):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean, std)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(mean, std)
        m.bias.data.fill_(0)

def calc_gradient_penalty(dis, real_data, fake_data):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    alpha = torch.rand(opt.batch_size, 1, 1)
    alpha = alpha.expand_as(real_data)

    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates.to(device)
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = dis(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),\
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.LAMBDA
    return gradient_penalty

def main():
    # 选择运行环境
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    #设置随机数种子
    opt.manualSeed = random.randint(1, 10000) 
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # 加载数据集
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    _ = datasets.MNIST(root='data', train=True, download=True, transform=None)
    train_data = MNISTLayoutDataset('data')
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)

    #固定随机点用于测试
    z_cls = torch.FloatTensor(torch.ones(opt.batch_size, opt.element_num, opt.cls_num))#类别都是1
    z_geo = torch.FloatTensor(opt.batch_size, opt.element_num, opt.geo_num).normal_(0.5, 0.15) #正态分布
    fixed_z = torch.cat((z_cls, z_geo), 2).to(device)

    # 加载模型
    print("load model")
    gen = model.Generator(opt.element_num, opt.geo_num, opt.cls_num).to(device)
    #gen = wmodel.Generator(opt.element_num, opt.geo_num, opt.cls_num).to(device)
    #dis = model.WifeDiscriminator(opt.batch_size).to(device)
    dis = model.RelationDiscriminator(opt.batch_size, opt.geo_num, opt.cls_num, opt.element_num).to(device)
    
    #模型初始化
    for layers in gen.modules():
        normal_init(layers, 0.0, 0.02)
    for layers in dis.modules():
        normal_init(layers, 0.0, 0.02)


    one = torch.FloatTensor([1])
    mone = one * -1
    one, mone = one.to(device), mone.to(device)

    # 定义优化器
    print("defined optimizers")
    g_optimizer = optim.Adam(gen.parameters(), opt.lrG, (opt.beta1, 0.9))
    d_optimizer = optim.Adam(dis.parameters(), opt.lrD, (opt.beta1, 0.9))
    #g_optimizer = optim.RMSprop(dis.parameters(), lr = opt.lrD)
    #d_optimizer = optim.RMSprop(gen.parameters(), lr = opt.lrG)

    # 设置为训练模式
    gen.train()
    dis.train()

    #结果文件夹
    if not os.path.isdir(opt.result):
        os.mkdir(opt.result)

    # 开始训练
    print('#######################################################')
    print('start train!!!')
    start_time = time.time()

    for epoch in range(opt.num_epochs):
        epoch_start_time = time.time()

        data_iter = iter(train_loader)

        for iter_d in range(opt.Diters):
            #训练判别器
            for p in dis.parameters():
                p.requires_grad = True
            #用real训练
            data = data_iter.next()
            real_images = torch.Tensor(data)
            real_images = real_images.to(device)
            real_images_v = Variable(real_images, requires_grad=True)

            dis.zero_grad()
            errD_real = dis(real_images)
            errD_real = errD_real.mean()
            errD_real.backward(mone)

            #用fake训练
            z_cls = torch.FloatTensor(torch.ones(opt.batch_size, opt.element_num, opt.cls_num))#类别都是1
            z_geo = torch.FloatTensor(opt.batch_size, opt.element_num, opt.geo_num).normal_(0.5, 0.15) #正态分布
            z = torch.cat((z_cls, z_geo), 2).to(device)
            z_v = Variable(z)
            fake_images_d = Variable(gen(z_v).data, requires_grad=True)
            input_v = fake_images_d
            errD_fake = dis(input_v)
            errD_fake = errD_fake.mean()
            errD_fake.backward(one)

            gradient_penalty = calc_gradient_penalty(dis, real_images_v.data, fake_images_d.data)
            gradient_penalty.backward()

            #计算判别器损失
            D_cost = errD_fake - errD_real + gradient_penalty
            d_optimizer.step()

            #训练生成器
            for p in dis.parameters():
                p.requires_grad = False

            gen.zero_grad()
            z_cls = torch.FloatTensor(torch.ones(opt.batch_size, opt.element_num, opt.cls_num))#类别都是1
            z_geo = torch.FloatTensor(opt.batch_size, opt.element_num, opt.geo_num).normal_(0.5, 0.15) #正态分布
            z = torch.cat((z_cls, z_geo), 2).to(device)
            z_v = Variable(z)

            fake_images_g = gen(z_v)
            errG = dis(fake_images_g)
            errG = errG.mean()
            errG.backward(mone)

            G_cost = -errG
            g_optimizer.step()

            print('[%d/%d] Loss_D: %.6f Loss_G: %.6f' % (epoch, opt.num_epochs, D_cost.item(), G_cost.item()))
            
            if epoch % 100 == 0:
                real_images = points_to_image(real_images[:64, :, :]).view(-1, 1, 28, 28)
                vutils.save_image(real_images, '{0}/real_samples.png'.format(opt.result), nrow=8)

                fake = gen(Variable(fixed_z))
                fake_images = points_to_image(fake[:64, :, :]).view(-1, 1, 28, 28)
                vutils.save_image(fake_images, '{0}/fake_samples_{1}.png'.format(opt.result, epoch), nrow=8)
    print('###########################################################')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=64, help='input batch size')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--element-num', type=int, default=128)
    parser.add_argument('--cls-num', type=int, default=1)
    parser.add_argument('--geo-num', type=int, default=2)
    parser.add_argument('--num-epochs', type=int, default=10000)
    parser.add_argument('--result', type=str, default='result_images')
    parser.add_argument('--Diters', type=int, default=5)
    parser.add_argument('--LAMBDA', type=int, default=10)

    opt = parser.parse_args()
    print(opt)
    main()