import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        size = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(size, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(size, -1).data, v.data))
        sigma = u.dot(w.view(size, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        self.bn1 = nn.BatchNorm2d(256)
        self.conv3 = SpectralNorm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1))
        self.bn2 = nn.BatchNorm2d(512)
        self.conv4 = SpectralNorm(nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1))
        self.bn3 = nn.BatchNorm2d(1024)
        self.conv5 = SpectralNorm(nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=1))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.bn1(self.conv2(x)))
        x = self.leaky_relu(self.bn2(self.conv3(x)))
        x = self.leaky_relu(self.bn3(self.conv4(x)))
        x = self.conv5(x)
        return x

class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.conv1 = nn.ConvTranspose2d(self.noise_dim, 1024, kernel_size=4, stride=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.tanh(self.conv5(x))
        return x
