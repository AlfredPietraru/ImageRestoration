import torch
import torch.linalg
import preprocessing as prep
from net import Encoder, Decoder
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
SIZE = 512
LATENT = 128
BATCH_SIZE = 10

def reparametrization_technique(averages, variances):
    e_values = torch.randn(size=[BATCH_SIZE, LATENT, LATENT])
    return e_values * variances + averages.unsqueeze(dim = -1)  


def KL_divergence_one_value(avg : torch.Tensor, var : torch.Tensor):
    return torch.trace(var) + torch.pow(torch.norm(avg), 2) - \
            LATENT - torch.log(torch.linalg.det(var))

def ELBO(x, x_recon, avg, var):
    print(x_recon.shape)
    covariance_matrix = torch.stack([torch.eye(SIZE), torch.eye(SIZE), torch.eye(SIZE)], dim=0)
    print(covariance_matrix.shape)
    dist = MultivariateNormal(x_recon, covariance_matrix=torch.eye(SIZE))
    result = dist.log_prob(x)
    print(result)
    print(result.shape)


encoder = Encoder(batch_size=BATCH_SIZE)
decoder = Decoder(nr_channels=1)

X = prep.get_only_training_image(0, BATCH_SIZE)
averages, variances = encoder(X)
Z = reparametrization_technique(averages, variances)
X_recon = decoder(Z)
ELBO(X[0], X_recon[0], averages[0], variances[0])

