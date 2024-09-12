import math
import torch
import torch.nn as nn
ORIGINAL = 512
LATENT = 128

# https://pub.aimind.so/image-restoration-using-deep-learning-variational-autoencoders-8483135bb72d
class Encoder(nn.Module):
    def __init__(self, nr_channels = 3, batch_size = 10):
        super().__init__()
        self.nr_channels = nr_channels
        self.batch_size = batch_size
        self.encoder = nn.Sequential(
         nn.Conv2d(self.nr_channels, 32, kernel_size=4, stride=2, padding=1),
         nn.ReLU(),
         nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
         nn.ReLU(),
         nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
         nn.ReLU(),
         nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
         nn.ReLU(),
         nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
         nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
         nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
        )

        self.fc1 = nn.Linear(512 * 4 * 4, 512 * 4)
        self.get_average = nn.Linear(512 * 4, LATENT)
        self.get_log_variance = nn.Linear(512 * 4, LATENT * LATENT)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten() 

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(self.flatten(x))
        mu = self.get_average(x)
        variance = torch.exp(self.get_log_variance(x))
        variance = variance.reshape(shape=(self.batch_size, LATENT, LATENT))
        return mu, variance

class Decoder(nn.Module):
    def __init__(self, nr_channels = 1):
        super().__init__()
        self.nr_channels = nr_channels
        self.decoder = nn.Sequential(
        nn.ConvTranspose2d(nr_channels,out_channels=3, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.decoder(x)
        return x
    
class VAE(nn.Module):
    def __init__(self, batch_size = 10):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.batch_size = 10

    def reparametrization_technique(self, averages, variances):
        e_values = torch.randn(size=[self.batch_size, LATENT, LATENT])
        return e_values * variances + averages.unsqueeze(dim = -1)  

    def forward(self, x):
        mu, variance = self.encoder(x)
        z = self.reparametrization_technique(mu, variance)
        return self.decoder(z), mu, variance 


