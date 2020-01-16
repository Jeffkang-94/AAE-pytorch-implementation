# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import imageio
import os
import models
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.pyplot as plt
#from Discriminator import Discriminator
#from Generator import Generator
from torch.autograd.variable import Variable
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))
                ])
### default parameter ###
input_dim = 28*28
batch_size= 100
D_lr = 2e-4
en_lr = 2e-4
de_lr = 2e-4
AE_criterion=nn.MSELoss()
GAN_criterion = nn.BCELoss()
epochs= 1000
latent_z = 8 # 8 Dimension
hidden_dim = 512

assert torch.cuda.is_available()
device=torch.device("cuda:0")
torch.backends.cudnn.benchmark = True
discriminator = models.Discriminator(latent_z,hidden_dim).to(device)
encoder = models.Encoder(input_dim, latent_z, hidden_dim).to(device)
decoder = models.Decoder(input_dim,latent_z, hidden_dim).to(device)

### Optimizer Adam ###
d_optim = torch.optim.Adam(discriminator.parameters(),lr=D_lr)
en_optim = torch.optim.Adam(encoder.parameters(), lr=en_lr)
de_optim = torch.optim.Adam(decoder.parameters(), lr=de_lr)
en_reg_optim = torch.optim.Adam(encoder.parameters(), lr=en_lr) # Regularization 

### Training label, Real =1, Fake = 0 ###
D_real_label=torch.ones(batch_size,1, dtype=torch.float32).to(device)
D_fake_label=torch.zeros(batch_size,1,dtype=torch.float32).to(device)

trainset = MNIST(root='./data/', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
fixed_data,_ = next(iter(trainloader)) ## fixed noise to generate sample images

f_loss=open("loss.txt","w")

if not os.path.exists("./samples"):
    os.makedirs("./samples")

def get_noise(n, latent_z=latent_z):
    return Variable(torch.randn(n, latent_z)).to(device)

def train_discriminator(optimizer,n,real_data, fake_data):
    encoder.eval()
    optimizer.zero_grad()
    D_real_label = torch.ones(n,1,dtype=torch.float32).to(device)
    D_fake_label = torch.zeros(n,1, dtype=torch.float32).to(device)

    real_noise = get_noise(n)
    D_real_noise = discriminator(real_noise)

    fake_noise = encoder(real_data)
    D_fake_noise = discriminator(fake_noise)

    D_loss_real = GAN_criterion(D_real_noise,D_real_label)
    D_loss_fake = GAN_criterion(D_fake_noise,D_fake_label)
    D_loss = (D_loss_real + D_loss_fake) / 2
    D_loss.backward()
    optimizer.step()

    return D_loss

def train_generator(optimizer, real_data):
    encoder.train()
    optimizer.zero_grad()
    noise = encoder(real_data)
    D_loss_fake = discriminator(noise)
    G_loss = GAN_criterion(D_loss_fake, D_real_label)
    G_loss.backward()
    optimizer.step()
    return G_loss

def train_autoencoder(optimizers, real_data, fake_data):
    total_AE_loss =0
    for p in discriminator.parameters():
        p.requires_grad =False

    optimizers[0].zero_grad()
    optimizers[1].zero_grad()
    AE_loss = AE_criterion(fake_data,real_data) ## Reconstruction Loss
    AE_loss.backward()
    optimizers[0].step()
    optimizers[1].step()
    return AE_loss

def make_sample(epoch):
    real_data = Variable(fixed_data).to(device)
    real_data = real_data.view(-1,input_dim)
    encoding = encoder(real_data)
    fake_data = decoder(encoding)
    plt.figure(figsize=(10,10))
    plt.axis("off")
    img=fake_data[:100].detach().cpu()
    img=img.view(-1,1,28,28)
    plt.imshow(np.transpose(make_grid(img,nrow=10,normalize=True),(1,2,0)))
    plt.savefig("./samples/epoch_%d.png" %(epoch))
    plt.close('all')

def make_real():
    real_data = Variable(fixed_data).to(device)
    real_data = real_data.view(-1,input_dim)
    real_data = real_data[:100]
    plt.figure(figsize=(10,10))
    plt.axis("off")
    real_data=real_data.view(-1,1,28,28)
    real_data=real_data.detach().cpu()
    plt.imshow(np.transpose(make_grid(real_data,nrow=10,normalize=True),(1,2,0)))
    plt.savefig("./samples/real.png")
    plt.close("all")

def make_loss_plot(R_loss,D_loss,G_loss):
    plt.plot(R_loss, label='R_loss')
    plt.plot(D_loss, label='D_loss')
    plt.plot(G_loss, label='G_loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss.png')
    plt.close()

def train_aae(step,real_data,fake_data,n):
    R_loss = train_autoencoder([en_optim, de_optim], real_data,fake_data) # Reconstruction
    ## GAN ##
    D_loss = train_discriminator(d_optim,n, real_data, fake_data)
    G_loss = train_generator(en_reg_optim, real_data)
    if step % 10 ==0:
        print ('Step [%d], recon_loss: %.4f, discriminator_loss :%.4f , generator_loss:%.4f'
                        %(step, R_loss.item(), D_loss.item(), G_loss.item()))
        f_loss.write('%.4f/%.4f/%.4f\n'%(R_loss.item(), D_loss.item(), G_loss.item()))
        f_loss.flush()


######## Main #########
make_real() # Generate real image, p_data
for epoch in range(epochs):
    print('[%d/%d] Epoch ========================================================' %(epoch, epochs))
    make_sample(epoch)
    for i, data in enumerate(trainloader):
        for para in discriminator.parameters():
            para.requires_grad=False
        imgs, _ = data
        n = len(imgs)
        real_data = imgs.to(device)
        real_data = real_data.view(-1,input_dim)

        ## AE ##
        encoding = encoder(real_data)
        fake_data = decoder(encoding)

        ## Training AAE ##
        train_aae(i,real_data,fake_data,n)

f_loss.close()
