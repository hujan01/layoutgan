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
<<<<<<< HEAD
import torchvision.utils as vutils
from dataset import MNISTLayoutDataset
import model, wmodel
=======
from torchvision.utils import save_image

from dataset import MNISTLayoutDataset
import model
>>>>>>> 839a889a307cbdbb3f85f5e5980fca56e0cc8a54
def points_to_image(points):
    """ 坐标点转成像素图 """
    batch_size = points.size(0)
    images = []
    for b in range(batch_size):
        canvas = np.zeros((28, 28)) #生成背景图片
        image = points[b]  #第一张图片
        for point in image:
            if point[0] > 0: #看概率是否大于阈值
<<<<<<< HEAD
                x, y = int(point[1]), int(point[2])
=======
                x, y = int(point[1]*28), int(point[2]*28)
>>>>>>> 839a889a307cbdbb3f85f5e5980fca56e0cc8a54
                x, y = min(x, 27), min(y, 27)
                canvas[x, y] = 255
        images.append(canvas)
    images = np.asarray(images)
    images_tensor = torch.from_numpy(images)
    return images_tensor
<<<<<<< HEAD
    
=======
>>>>>>> 839a889a307cbdbb3f85f5e5980fca56e0cc8a54
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

    # 加载模型
    print("load model")
<<<<<<< HEAD
    #gen = model.Generator(opt.element_num, opt.geo_num, opt.cls_num).to(device)
    gen = wmodel.Generator(opt.element_num, opt.geo_num, opt.cls_num).to(device)
    dis = wmodel.Discriminator(opt.batch_size).to(device)
    #dis = model.Discriminator(opt.batch_size, opt.geo_num, opt.cls_num, opt.element_num).to(device)
=======
    gen = model.Generator(opt.element_num, opt.geo_num, opt.cls_num).to(device)
    dis = model.Discriminator(opt.batch_size, opt.geo_num, opt.cls_num, opt.element_num).to(device)
>>>>>>> 839a889a307cbdbb3f85f5e5980fca56e0cc8a54
    #初始化
    for layers in gen.modules():
        normal_init(layers, 0.0, 0.02)
    for layers in dis.modules():
        normal_init(layers, 0.0, 0.02)

    z_cls = torch.FloatTensor(torch.ones(opt.batch_size, opt.element_num, opt.cls_num))#类别都是1
    z_geo = torch.FloatTensor(opt.batch_size, opt.element_num, opt.geo_num).normal_(0.5, 0.15) #正态分布
    fixed_z = torch.cat((z_cls, z_geo), 2).to(device)

    one = torch.FloatTensor([1])
    mone = one * -1
    one, mone = one.to(device), mone.to(device)

    # 定义优化器
    print("Initialize optimizers")
    #g_optimizer = optim.Adam(gen.parameters(), opt.lrG, (opt.beta1, 0.9))
    #d_optimizer = optim.Adam(dis.parameters(), opt.lrD, (opt.beta1, 0.9))
    g_optimizer = optim.RMSprop(dis.parameters(), lr = opt.lrD)
    d_optimizer = optim.RMSprop(gen.parameters(), lr = opt.lrG)

    # 设置为训练模式
    gen.train()
    dis.train()

<<<<<<< HEAD
    if not os.path.isdir(opt.result):
        os.mkdir(opt.result)
=======
>>>>>>> 839a889a307cbdbb3f85f5e5980fca56e0cc8a54
    # 开始训练
    print('#######################################################')
    print('start train!!!')
    start_time = time.time()
    gen_iterations = 0 #生成器迭代次数
    for epoch in range(opt.num_epochs):
        epoch_start_time = time.time()
        data_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
        
            for p in dis.parameters():
                p.requires_grad = True

            if gen_iterations<25 or gen_iterations%500 == 0:
                Diters = 100
            else:
                Diters = 5 #每迭代5次判别器，迭代一次生成器

            j = 0
            while j<Diters and i<len(train_loader):
                j += 1
                for p in dis.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                i += 1
                data = data_iter.next() #64, 128, 3
                dis.zero_grad()
                #用real训练
                real_images = data
                real_images = real_images.to(device)
                real_images = Variable(real_images) 
                batch_size = real_images.size(0)

                errD_real = dis(real_images)
                errD_real.backward(one)
                # 用fake训练
                z_cls = torch.FloatTensor(torch.ones(opt.batch_size, opt.element_num, opt.cls_num))#类别都是1
                z_geo = torch.FloatTensor(opt.batch_size, opt.element_num, opt.geo_num).normal_(0.5, 0.15) #正态分布
                z = torch.cat((z_cls, z_geo), 2).to(device)
                z_v = Variable(z, volatile = True)
                fake_images_d = Variable(gen(z_v).data)
                errD_fake = dis(fake_images_d)
                errD_fake.backward(mone)
                errD = errD_real - errD_fake
                d_optimizer.step()

            #训练生成器
            for p in dis.parameters():
                p.requires_grad = False

            gen.zero_grad()
            z_cls = torch.FloatTensor(torch.ones(opt.batch_size, opt.element_num, opt.cls_num))#类别都是1
            z_geo = torch.FloatTensor(opt.batch_size, opt.element_num, opt.geo_num).normal_(0.5, 0.15) #正态分布
            z = torch.cat((z_cls, z_geo), 2).to(device)
            fake_images_g = gen(z)
            errG = dis(fake_images_g)
            errG.backward(one)
            g_optimizer.step()
            gen_iterations += 1

            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                % (epoch, opt.num_epochs, i, len(train_data), gen_iterations,\
                    errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
            
            if gen_iterations % 500 == 0:
<<<<<<< HEAD
                real_images = points_to_image(real_images).view(-1, 1, 28, 28)
                vutils.save_image(real_images, '{0}/real_samples.png'.format(opt.result))

                fake = gen(Variable(fixed_z, volatile=True))
                fake[:, :, 1:] *= 28
                fake_images = points_to_image(fake).view(-1, 1, 28, 28)
                vutils.save_image(fake_images, '{0}/fake_samples_{1}.png'.format(opt.result, gen_iterations), nrow=8)
=======
                real_images = points_to_image(real_images)
                vutils.save_image(real_images, '{0}/real_samples.png'.format(opt.result))

                fake = gen(Variable(fixed_z, volatile=True))
                fake_images = points_to_image(fake)
                vutils.save_image(fake_images, '{0}/fake_samples_{1}.png'.format(opt.result, gen_iterations))
>>>>>>> 839a889a307cbdbb3f85f5e5980fca56e0cc8a54
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
<<<<<<< HEAD
    parser.add_argument('--num-epochs', type=int, default=100)
=======
    parser.add_argument('--num-epochs', type=int, default=10)
>>>>>>> 839a889a307cbdbb3f85f5e5980fca56e0cc8a54
    parser.add_argument('--result', type=str, default='result_images')

    opt = parser.parse_args()
    print(opt)
    main()