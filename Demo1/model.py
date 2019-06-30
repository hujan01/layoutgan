import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# attention
def relation_module(out, unary, psi, phi, wr):
    element_num = out.size(1)  
    batch_res = []
    for bdx, batch in enumerate(out):
        f_prime = []
        # i, j are two elements.
        for idx, i in enumerate(batch):
            self_attention = torch.Tensor(torch.zeros(i.size(0))).to('cuda')
            for jdx, j in enumerate(batch):
                if idx == jdx:
                    continue
                u = F.relu(unary(j))
                iv = i.view(i.size(0), 1)
                jv = j.view(j.size(0), 1)
                dot = (torch.mm((iv * psi).t(), jv * phi)).squeeze()
                self_attention += dot * u
            f_prime.append(wr * (self_attention / element_num) + i)
        batch_res.append(torch.stack(f_prime))
    return torch.stack(batch_res)


class Generator(nn.Module):
    """The generator (in GAN)"""

    def __init__(self, batch_size, element_num, geo_num, cls_num):
        super(Generator, self).__init__()
        self.feature_size = geo_num + cls_num #特征维度
        self.cls_num = cls_num 
        self.geo_num = geo_num 
        self.element_num = element_num

        # Encoder
        self.encoder_fc1 = nn.Linear(self.feature_size, self.feature_size * 2)
        self.encoder_batch_norm1 = nn.BatchNorm1d(element_num)
        self.encoder_fc2 = nn.Linear(self.feature_size * 2, self.feature_size * 2 * 2)
        self.encoder_batch_norm2 = nn.BatchNorm1d(element_num)
        self.encoder_fc3 = nn.Linear(self.feature_size * 2 * 2, self.feature_size * 2 * 2)

        # Relation model 1
        self.relation1_unary = nn.Linear(self.feature_size * 2 * 2,
                                         self.feature_size * 2 * 2)  
        self.relation1_psi = torch.Tensor(torch.rand(1)).to('cuda')   
        self.relation1_phi = torch.Tensor(torch.rand(1)).to('cuda')   
        self.relation1_wr = torch.Tensor(torch.rand(1)).to('cuda')   

        # Relation model 2
        self.relation2_unary = nn.Linear(self.feature_size * 2 * 2,
                                         self.feature_size * 2 * 2)  
        self.relation2_psi = torch.Tensor(torch.rand(1)).to('cuda')  
        self.relation2_phi = torch.Tensor(torch.rand(1)).to('cuda')   
        self.relation2_wr = torch.Tensor(torch.rand(1)).to('cuda') 

        # Relation model 3
        self.relation3_unary = nn.Linear(self.feature_size * 2 * 2,
                                         self.feature_size * 2 * 2)  
        self.relation3_psi = torch.Tensor(torch.rand(1)).to('cuda')   
        self.relation3_phi = torch.Tensor(torch.rand(1)).to('cuda')  
        self.relation3_wr = torch.Tensor(torch.rand(1)).to('cuda')  

        # Relation model 4
        self.relation4_unary = nn.Linear(self.feature_size * 2 * 2,
                                         self.feature_size * 2 * 2)  
        self.relation4_psi = torch.Tensor(torch.rand(1)).to('cuda')   
        self.relation4_phi = torch.Tensor(torch.rand(1)).to('cuda')   
        self.relation4_wr = torch.Tensor(torch.rand(1)).to('cuda')  

        # Decoder
        self.decoder_fc1 = nn.Linear(self.feature_size * 2 * 2, self.feature_size * 2)
        self.decoder_batch_norm1 = nn.BatchNorm1d(element_num)
        self.decoder_fc2 = nn.Linear(self.feature_size * 2, self.feature_size)

        # Branch
        self.branch_fc1 = nn.Linear(self.feature_size, cls_num)
        self.branch_fc2 = nn.Linear(self.feature_size, geo_num)

    def forward(self, input):
        # Encoder
        out = F.relu(self.encoder_batch_norm1(self.encoder_fc1(input)))
        out = F.relu(self.encoder_batch_norm2(self.encoder_fc2(out)))
        encoded = torch.sigmoid(self.encoder_fc3(out))

        # Stacked relation module
        relation_residual_1 = relation_module(encoded, self.relation1_unary, self.relation1_psi,
                                              self.relation1_phi, self.relation1_wr)
        relation_residual_2 = relation_module(relation_residual_1, self.relation2_unary, self.relation2_psi,
                                              self.relation2_phi, self.relation2_wr)
        relation_residual_3 = relation_module(relation_residual_2, self.relation3_unary, self.relation3_psi,
                                              self.relation3_phi, self.relation3_wr)
        relation_residual_4 = relation_module(relation_residual_3, self.relation4_unary, self.relation4_psi,
                                              self.relation4_phi, self.relation4_wr)

        # Decoder
        out = F.relu(self.decoder_batch_norm1(self.decoder_fc1(relation_residual_4)))
        out = F.relu(self.decoder_fc2(out))

        # Branch
        syn_cls = self.branch_fc1(out)
        syn_geo = self.branch_fc2(out)

        # Synthesized layout
        res = torch.cat((syn_cls, syn_geo), 2)
        return res

#判别器
class Discriminator(nn.Module):
    def __init__(self, batch_size):
        super(Discriminator, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        # Convolution Layers
        self.conv1 = nn.Conv2d(1, 4, 3, 1, 1)
        torch.nn.init.normal_(self.conv1.weight, 0, 0.02)
        self.conv1_bn = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3, 1, 1)
        torch.nn.init.normal_(self.conv2.weight, 0, 0.02)
        self.conv2_bn = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3, 1, 1)
        torch.nn.init.normal_(self.conv3.weight, 0, 0.02)
        self.conv3_bn = nn.BatchNorm2d(16)

        # Fully Connected Layers
        self.fc1 = nn.Linear(16*7*7, 128)
        torch.nn.init.normal_(self.fc1.weight, 0, 0.02)
        self.fc2 = nn.Linear(128, 1)
        torch.nn.init.normal_(self.fc2.weight, 0, 0.02)


    def forward(self, x_in):
        # Passing through wireframe rendering
        x_wf = self.wireframe_rendering(x_in)

        # Passing through conv layers
        x = torch.nn.functional.max_pool2d(F.relu(self.conv1_bn(self.conv1(x_wf))), 2, 2)
        x = torch.nn.functional.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), 2, 2)
        x = F.relu(self.conv3_bn(self.conv3(x)))

        # Flattening and passing through FC Layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x

    def wireframe_rendering(self, x_in):
        def k(x):
            return torch.relu(1-torch.abs(x))

        w = 28
        h = 28

        p = x_in[:, :, 0]
        theta = x_in[:, :, 1:]

        batch_size, num_elements, geo_size = theta.shape

        theta[:, :, 0] *= w
        theta[:, :, 1] *= h

        assert(p.shape[0] == batch_size and p.shape[1] == num_elements)

        x = np.repeat(np.arange(w), h).reshape(w,h)
        y = np.transpose(x)

        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y)
        x_tensor = x_tensor.view(1, w, h)
        y_tensor = y_tensor.view(1, w, h)

        base_tensor = torch.cat([x_tensor, y_tensor]).type(torch.FloatTensor).to(self.device)
        base_tensor = base_tensor.repeat(batch_size*num_elements, 1, 1, 1)
        theta= theta.view(batch_size*num_elements, geo_size, 1, 1)
        p = p.view(batch_size, num_elements, 1, 1)

        F = k(base_tensor[:,0,:,:] - theta[:,0]) * k(base_tensor[:, 1, :, :] - theta[:, 1])
        F = F.view(batch_size, num_elements, w, h)

        p_times_F = p * F

        I = torch.max(p_times_F, dim=1)[0]

        I = I.view(batch_size, 1, w, h)
        return I
