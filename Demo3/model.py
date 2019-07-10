""" 添加attention模块 """
import os
import random

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter


<<<<<<< HEAD
=======

>>>>>>> 839a889a307cbdbb3f85f5e5980fca56e0cc8a54
class Attention(nn.Module):
    """ attention model """
    def __init__(self, in_channels, out_channels=None, dimension=1, sub_sample=False, bn=True, generate=True):
        super(Attention, self).__init__()
        if out_channels is None:
            self.out_channels = in_channels//2 if in_channels>1 else 1
        self.out_channels = out_channels
        self.generate = generate #是否加入残差
        self.g = nn.Conv1d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0) #U
        
        self.theta = nn.Conv1d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.theta.weight, 0, 0.02)
        self.phi = nn.Conv1d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.phi.weight, 0, 0.02)        
        self.W = nn.Sequential(nn.Conv1d(self.out_channels, in_channels, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm1d(in_channels))
        #nn.init.constant(self.W[1].weight, 0) 
        nn.init.normal_(self.W[1].weight, 0, 0.02)
        nn.init.constant(self.W[1].bias, 0)
        if sub_sample: #是否需要下采样，这里会用到最大池化
            self.g=nn.Sequential(self.g, nn.MaxPool1d)
            self.phi=nn.Sequential(self.phi, nn.MaxPool1d)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x): #x: (256, 128, 12)
        batch_size = x.size(0) #批次大小
        g_x = self.g(x).view(batch_size, self.out_channels, -1) 
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.out_channels, -1)  
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.out_channels, -1)
        f = torch.matmul(theta_x, phi_x) #计算H 
 
        N = f.size(-1)
        f_div_c = f/N
        y = torch.matmul(f_div_c, g_x)
        y = y.permute(0,2,1).contiguous()
        #y = y.view(batch_size, self.out_channels, *x.size()[2:])
        W_y = self.W(y)
        if self.generate: 
            output = W_y + x
        else:
            output=W_y
        return output

class Generator(nn.Module):
    def __init__(self, num_elements, geo_num, cls_num): #位置个数，元素个数
        super(Generator, self).__init__()
        self.geo_num = geo_num
        self.cls_num = cls_num
        self.feature_size = geo_num + cls_num
        # Encode
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size*2),
            nn.ReLU(True),
            nn.Linear(self.feature_size*2, self.feature_size*2*2),
            nn.ReLU(True),
            nn.Linear(self.feature_size*2*2, self.feature_size*2*2)
        )

        # Attention
        self.attention = nn.Sequential(
<<<<<<< HEAD
            Attention(num_elements, 1, generate=True),
            Attention(num_elements, 1, generate=True),
            Attention(num_elements, 1, generate=True),
            Attention(num_elements, 1, generate=True)
=======
            Attention(self.feature_size*2*2, 1, generate=True),
            Attention(self.feature_size*2*2, 1, generate=True),
            Attention(self.feature_size*2*2, 1, generate=True),
            Attention(self.feature_size*2*2, 1, generate=True)
>>>>>>> 839a889a307cbdbb3f85f5e5980fca56e0cc8a54
        )   
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_size*2*2, self.feature_size*2),
            nn.ReLU(True),
            nn.Linear(self.feature_size*2, self.feature_size) 
        )    

        #branch
        self.fc6 = nn.Linear(self.feature_size, cls_num)
        self.fc7 = nn.Linear(self.feature_size, geo_num)

            
    def forward(self, x_in):

        x = self.encoder(x_in)
<<<<<<< HEAD
        x = self.attention(x)
=======
        x = x.permute(0, 2, 1).contiguous()
        x = self.attention(x)
        x = x.permute(0, 2, 1).contiguous() #维度变换后，使用该函数，方可view对维度进行变形
>>>>>>> 839a889a307cbdbb3f85f5e5980fca56e0cc8a54
        x = self.decoder(x)
        cls = torch.sigmoid(self.fc6(x))
        #cls = torch.nn.LeakyReLU(self.fc6(out))
        #cls = torch.relu(self.fc6(out))
        geo = torch.sigmoid(self.fc7(x))
        #geo = torch.nn.LeakyReLU(self.fc7(out))
        #geo = torch.relu(self.fc7(out))
        output = torch.cat((cls, geo), 2)
        return output

#判别器
class Discriminator(nn.Module):
    """ relation_based """
    def __init__(self, batch_size, geo_num, cls_num, num_elements):
        super(Discriminator, self).__init__()
        self.batch_size = batch_size
        self.geo_num = geo_num
        self.cls_num = cls_num
        self.feature_size = geo_num + cls_num

        # Encode
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size*2),
            nn.BatchNorm1d(num_elements),
            nn.ReLU(True),
            nn.Linear(self.feature_size*2, self.feature_size*2*2),
            nn.BatchNorm1d(num_elements),
            nn.ReLU(True),
            nn.Linear(self.feature_size*2*2, self.feature_size*2*2)
        )

        # attention
        self.attention = nn.Sequential(
<<<<<<< HEAD
            Attention(num_elements, 1, generate=False),
            Attention(num_elements, 1, generate=False),
            Attention(num_elements, 1, generate=False),
            Attention(num_elements, 1, generate=False)
=======
            Attention(self.feature_size*2*2, 1, generate=False),
            Attention(self.feature_size*2*2, 1, generate=False),
            Attention(self.feature_size*2*2, 1, generate=False),
            Attention(self.feature_size*2*2, 1, generate=False)
>>>>>>> 839a889a307cbdbb3f85f5e5980fca56e0cc8a54
        )      

        #max-pooling 
        self.g = nn.MaxPool1d(kernel_size=num_elements)

        #Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_size*2*2, self.feature_size*2),
            nn.ReLU(True),
            nn.Linear(self.feature_size*2, 1) 
<<<<<<< HEAD
        )
            
    def forward(self, x_in):
        x = self.encoder(x_in)
        x = self.attention(x)
        x = self.decoder(x)
        x = x.mean(1)
        return x.mean(0)
=======
        )    
    def forward(self, x_in):
        x = self.encoder(x_in)
        x = x.permute(0, 2, 1).contiguous()
        x = self.attention(x)
        x = self.g(x).permute(0, 2, 1).contiguous()
        x = self.decoder(x).view(-1, 1)
        x = x.mean(0)
        return x.view(1)
>>>>>>> 839a889a307cbdbb3f85f5e5980fca56e0cc8a54
