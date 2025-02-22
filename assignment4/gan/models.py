import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1024, 4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),

            nn.Conv2d(1024, 1, 4, stride=1, padding=0),
        )

    def forward(self, x):
        # return self.main(x) 
        x = self.main(x)
        return x.view(-1)
        
class Generator(nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.output_channels = output_channels
        self.main = nn.Sequential(
            
            nn.ConvTranspose2d(self.noise_dim, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, self.output_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, self.noise_dim, 1, 1)
        x = self.main(x)
        return x

