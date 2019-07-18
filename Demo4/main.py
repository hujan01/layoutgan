import torch
import torch.optim as optim
from dataloaders import get_mnist_dataloaders, get_lsun_dataloader
from models import Generator, RelationDiscriminator
from training import Trainer 
from torchvision import datasets , transforms
from dataset import  MNISTLayoutDataset
from torch.utils.data import DataLoader


#加载数据
_ = datasets.MNIST(root='data', train=True, download=True, transform=None)
train_data = MNISTLayoutDataset('data')
data_loader = DataLoader(train_data, batch_size=64, shuffle=True)

#定义模型
generator = Generator(128, 2, 1)
discriminator = RelationDiscriminator(64, 2, 1, 128) 

print(generator)
print(discriminator)

#定义优化器
lr = 1e-4
betas = (.5, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr/10, betas=betas)

#训练模型
epochs = 100
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available())
trainer.train(data_loader, epochs, save_training_gif=True)

#保存模型
name = 'mnist_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')
