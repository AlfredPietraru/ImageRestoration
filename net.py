import math
import torch
import torch.nn as nn
SIZE = 512

class AutoEncoder(nn.Module):
    def __init__(self, nr_channels = 3):
        super().__init__()
        self.nr_channels = nr_channels
        self.enc_1 = nn.Sequential(
          nn.Conv2d(nr_channels, nr_channels, kernel_size=3, stride=1, padding=1),
          nn.Conv2d(nr_channels, nr_channels, kernel_size=3, stride=1, padding=1),  
        )
        self.enc_2 = nn.Sequential(
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
          nn.Conv2d(nr_channels, nr_channels, kernel_size=3, stride=1, padding=1),
          nn.Conv2d(nr_channels, nr_channels, kernel_size=3, stride=1, padding=1),
        )
        self.enc_3 = nn.Sequential(
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
          nn.Conv2d(nr_channels, nr_channels, kernel_size=3, stride=1, padding=1),
          nn.Conv2d(nr_channels, nr_channels, kernel_size=3, stride=1, padding=1),
        )
        self.enc_4 = nn.Sequential(
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
          nn.Conv2d(nr_channels, 16, kernel_size=3, stride=1, padding=1),
          nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        )
        self.enc_5 = nn.Sequential(
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
          nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
          nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

        self.enc_6 = nn.Sequential(
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
          nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
          nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  
        )

        self.dec_1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
        )
        self.dec_3 = nn.Sequential(
          nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
          nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
          nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
        )

        self.dec_4 = nn.Sequential(
            nn.Conv2d(19, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(16, nr_channels, kernel_size=4, stride=2, padding=1)
        )

        self.dec_5 = nn.Sequential(
            nn.Conv2d(nr_channels * 2, nr_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(nr_channels, nr_channels, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(nr_channels, nr_channels, kernel_size=4, stride=2, padding=1)   
        )

        self.dec_6 = nn.Sequential(
          nn.Conv2d(nr_channels * 2, nr_channels, kernel_size=3, stride=1, padding=1),
          nn.Conv2d(nr_channels, nr_channels, kernel_size=3, stride=1, padding=1),
          nn.Conv2d(nr_channels, nr_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x : torch.Tensor):
        x1 = self.enc_1(x)
        x2 = self.enc_2(x1)
        x3 = self.enc_3(x2)
        x4 = self.enc_4(x3)
        x5 = self.enc_5(x4)
        x = self.enc_6(x5)
        x = self.dec_1(x)
        x = torch.cat([x, x5], dim=0)
        x = self.dec_2(x)
        x = torch.cat([x, x4], dim=0)
        x = self.dec_3(x)
        x = torch.cat([x, x3], dim=0)
        x = self.dec_4(x)
        x = torch.cat([x, x2], dim=0)
        x = self.dec_5(x)
        x = torch.cat([x, x1], dim=0)
        x = self.dec_6(x)
        return x