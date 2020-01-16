# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.pyplot as plt
from Discriminator import Discriminator
from Generator import Generator
from torch.autograd.variable import Variable
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
                ])
assert torch.cuda.is_available()
device=torch.device("cuda:0")
generator = Generator().to(device)
discriminator = Discriminator().to(device)
### default parameter ###
batch_size=100
D_lr = 2e-4
G_lr = 2e-4
criterion = nn.BCELoss()
epochs= 1000

### Optimizer Adam ###
g_optim = torch.optim.Adam(generator.parameters(), lr=D_lr)
d_optim = torch.optim.Adam(discriminator.parameters(), lr=G_lr)

### Training label, Real =1, Fake = 0 ###
D_real_label=torch.ones(batch_size,1, dtype=torch.float32).to(device)
D_fake_label=torch.zeros(batch_size,1,dtype=torch.float32).to(device)

trainset = MNIST(root='./data/', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)


g_losses = []
d_losses = []

def get_noise(n, latent_z=128):
    return Variable(torch.randn(n, latent_z)).to(device)

def train_discriminator(optimizer, real_data, fake_data):
    optimizer.zero_grad()
    D_loss_real = discriminator(real_data)
    error_real = criterion(D_loss_real, D_real_label)
    error_real.backward()

    D_loss_fake = discriminator(fake_data)
    error_fake = criterion(D_loss_fake, D_fake_label)
    error_fake.backward()
    optimizer.step()
    return (D_loss_real+ D_loss_fake)/2

def train_generator(optimizer, fake_data):
    optimizer.zero_grad()
    D_loss_fake = discriminator(fake_data)
    error = criterion(D_loss_fake, D_real_label)
    error.backward()
    optimizer.step()
    return D_loss_fake

def make_loss_plot(D_loss,G_loss):
    plt.plot(D_loss, label='D_loss')
    plt.plot(G_loss, label='G_loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss.png')
    plt.close()

samples = get_noise(100)
real,_=next(iter(trainloader))
real = make_grid(real, nrow=10, normalize=True)
real = np.transpose(real,(1,2,0))
plt.axis('off')
plt.imshow(real)
if not os.path.exists("./samples"):
    os.makedirs("./samples")
plt.tight_layout()
plt.savefig("./samples/real.png")
plt.close()


generator.train()
discriminator.train()
f_loss=open("loss.txt","w")
for epoch in range(epochs):
    D_loss=0
    G_loss=0
    for i, data in enumerate(trainloader):
        imgs, _ = data
        n = len(imgs)
        fake_data = generator(get_noise(n)).detach()
        real_data = imgs.to(device)
        D_loss += train_discriminator(d_optim, real_data, fake_data)

        fake_data = generator(get_noise(n))
        G_loss += train_generator(g_optim, fake_data)

    img = generator(samples).cpu().detach()
    img = make_grid(img, nrow=10, normalize=True)
    img = np.transpose(img,(1,2,0))
    plt.axis("off")
    plt.imshow(img)
    plt.tight_layout()
    plt.savefig("./samples/epoch_%d.png" %(epoch+1))
    plt.close()

    
    print('[%d/%d] G_loss : %.5f 	D_loss : %.5f' %(epoch,epochs, G_loss.mean()/i, D_loss.mean()/i))
    f_loss.write("%.5f/%.5f\n" %(G_loss.mean()/i, D_loss.mean()/i))
    f_loss.flush()
    #make_loss_plot(d_losses,g_losses)
f_loss.close()
