import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),  # (B, 64, 16, 16) assuming input is (32, 32)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, stride=1, padding=0),  # Final output (B, 1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)

class Generator(nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 256, 4, stride=1, padding=0),  # (B, 256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # (B, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, output_channels, 4, stride=2, padding=1),  # (B, 3, 32, 32)
            nn.Tanh()  # Normalize the images to [-1, 1]
        )

    def forward(self, x):
        x = x.view(-1, self.noise_dim, 1, 1)  # Reshape input noise vector into a mini-batch of inputs
        return self.net(x)
