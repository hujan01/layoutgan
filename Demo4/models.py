import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random

from matplotlib import pyplot as plt 

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

class Attention(nn.Module):
    """ attention model """
    def __init__(self, in_channels, out_channels=None, dimension=1, sub_sample=False, bn=True, generate=True):
        super(Attention, self).__init__()
        if out_channels is None:
            self.out_channels = in_channels//2 if in_channels>1 else 1
        #self.out_channels = out_channels
        self.generate = generate #是否加入残差
        self.g = nn.Conv1d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0) #U
        
        self.theta = nn.Conv1d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.theta.weight, 0, 0.02)
        self.phi = nn.Conv1d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.phi.weight, 0, 0.02)        
        self.W = nn.Sequential(nn.Conv1d(self.out_channels, in_channels, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm1d(in_channels))
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)
        if sub_sample: 
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
            nn.Linear(3, 128),
            #nn.BatchNorm1d(num_elements),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            #nn.BatchNorm1d(num_elements),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512)
        )

        # Attention
        self.attention_1 = Attention(512, generate=False)
        self.attention_2 = Attention(512, generate=False)
        self.attention_3 = Attention(512, generate=False)
        self.attention_4 = Attention(512, generate=False)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(num_elements),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(num_elements),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 3) 
        )    

        #branch
        self.fc6 = nn.Linear(3, cls_num)
        self.fc7 = nn.Linear(3, geo_num)
            
    def forward(self, x_in):
        x = self.encoder(x_in)
        x = x.permute(0, 2, 1).contiguous()
        x_hat = x.clone()
        x = self.attention_1(x)
        x = self.attention_2(x) + x_hat
        x_hat =x.clone()
        x = self.attention_3(x)
        x = self.attention_4(x) + x_hat
        x = x.permute(0, 2, 1).contiguous() #维度变换后，使用该函数，方可view对维度进行变形
        x = self.decoder(x)
        cls = torch.sigmoid(self.fc6(x))
        geo = torch.sigmoid(self.fc7(x))
        output = torch.cat((cls, geo), 2)
        return output

    def sample_latent(self, num_samples):
        z_cls = torch.FloatTensor(torch.ones(num_samples, 128, 1))#类别都是1
        z_geo = torch.FloatTensor(num_samples, 128, 2).normal_(0.5, 0.15) #正态分布
        fixed_z = torch.cat((z_cls, z_geo), 2)  
        return fixed_z   

class RelationDiscriminator(nn.Module):
    """ relation_based """
    def __init__(self, batch_size, geo_num, cls_num, num_elements):
        super(RelationDiscriminator, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.geo_num = geo_num
        self.cls_num = cls_num
        self.feature_size = geo_num + cls_num

        # Encode
        self.activatation = nn.LeakyReLU(0.1)
        self.encoder_fc1 = nn.Linear(3, 6)
        self.encoder_bn1 = nn.LayerNorm((num_elements, 6))   
        self.encoder_fc2 = nn.Linear(6, 12)
        self.encoder_bn2 = nn.LayerNorm((num_elements, 12)) 
        self.encoder_fc3 = nn.Linear(12, 24)

        # relation
        self.attention= Attention(24,  generate=False)
        
        #max-pooling 用于进行全局
        self.g = nn.MaxPool1d(kernel_size=num_elements)

        # Decode
        self.decoder_fc4 = nn.Linear(24, 12)
        self.decoder_fc5 = nn.Linear(12, 1)
        #self.decoder_fc6 = nn.Linear(self.feature_size*2, 1)

    def forward(self, x_in):
        
        x = self.activatation(self.encoder_bn1(self.encoder_fc1(x_in)))
        #x = self.activatation(self.encoder_fc1(x_in))
        x = self.activatation(self.encoder_bn2(self.encoder_fc2(x)))
        #x = self.activatation(self.encoder_fc2(x))
        x = self.encoder_fc3(x)

        x = x.permute(0,2,1)
        x = self.attention(x)

        x = self.g(x).permute(0, 2, 1)
        x = self.activatation(self.decoder_fc4(x))
        #x = self.activatation(self.decoder_fc5(x))
        x = self.decoder_fc5(x)
        return x.view(-1)

