import torch
from torch import nn
import torch.functional as F

class Encoder(nn.Module):
    def __init__(self, dim_input , dim_z, dim_h):
        super(Encoder, self).__init__()
        self.dim_input = dim_input
        self.dim_z = dim_z
        self.dim_h = dim_h

        self.encoder0 = nn.Sequential(
                         nn.Linear(self.dim_input,self.dim_h),
                         nn.ReLU()
                         )
        self.encoder1 = nn.Sequential(
                         nn.Linear(self.dim_h, self.dim_h//2),
                         nn.ReLU()
                         )
        self.encoder2 = nn.Sequential(
                         nn.Linear(self.dim_h//2,self.dim_z),
                         nn.ReLU()
                         )
    def forward(self, x):
        x = self.encoder0(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dim_input , dim_z,dim_h):
        super(Decoder, self).__init__()
        self.dim_input = dim_input
        self.dim_z = dim_z
        self.dim_h = dim_h
        self.decoder0 = nn.Sequential(
                          nn.Linear(self.dim_z, self.dim_h),
                          nn.ReLU()
                          )
        self.decoder1 = nn.Sequential(
                          nn.Linear(self.dim_h, self.dim_h*2),
                          nn.ReLU()
                          )
        self.decoder2 = nn.Sequential(
                          nn.Linear(self.dim_h*2, self.dim_input),
                          nn.ReLU()
                          )


    def forward(self, x):
        x = self.decoder0(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, dim_z , dim_h):
        super(Discriminator,self).__init__()
        self.dim_z = dim_z
        self.dim_h = dim_h
        self.network = []
        self.network.extend([
            nn.Linear(self.dim_z, self.dim_h),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.dim_h, self.dim_h//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.dim_h//2,1),
            nn.Sigmoid(),
        ])
        self.network = nn.Sequential(*self.network)

    def forward(self, z):
        disc = self.network(z)
        return disc
