import torch
import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_in = 784
        self.n_out = 1
        self.fc0 = nn.Sequential(
                    nn.Linear(self.n_in, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3)
                    )
#        self.fc1 = nn.Sequential(
#                    nn.Linear(1024, 512),
#                    nn.LeakyReLU(0.2),
#                    nn.Dropout(0.3)
#                    )
#        self.fc2 = nn.Sequential(
#                    nn.Linear(512, 256),
#                    nn.LeakyReLU(0.2),
#                    nn.Dropout(0.3)
#                    )
        self.fc3 = nn.Sequential(
                    nn.Linear(256, self.n_out),
                    nn.Sigmoid()
                    )
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc0(x)
#        x = self.fc1(x)
#        x = self.fc2(x)
        x = self.fc3(x)
        return x
