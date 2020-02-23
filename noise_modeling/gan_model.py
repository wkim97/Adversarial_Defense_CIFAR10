from __future__ import print_function
import torch.nn as nn

#############################################################################################################
# Generator
#############################################################################################################
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=32 * 8,
                               kernel_size=8, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(num_features=32 * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32 * 8, out_channels=32 * 4,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=32 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32 * 4, out_channels=3,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.Tanh())

    def forward(self, inputs):
        inputs = inputs.view(-1, 100, 1, 1)
        return self.main(inputs)

#############################################################################################################
# Implementation - discriminator
# Takes an image as input and outputs a scalar probability of whether it is real or fake
# Series of Conv2d, BatchNorm2d, and LeakyReLU
#############################################################################################################
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32 * 4,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=32 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32 * 4, out_channels=32 * 8,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=32 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32 * 8, out_channels=1,
                      kernel_size=8, stride=1, padding=0,
                      bias=False),
            nn.Sigmoid())

    def forward(self, inputs):
        o = self.main(inputs)
        return o.view(-1, 1)