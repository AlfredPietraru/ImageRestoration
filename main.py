import torch
import preprocessing as prep
from net import Encoder 
ORIGINAL = 512
LATENT = 128
BATCH_SIZE = 10

def reparametrization_technique(averages, variances):
    e_values = torch.randn(size=[BATCH_SIZE, LATENT, LATENT])
    return e_values * variances + averages.unsqueeze(dim = -1)

images = prep.get_only_training_image(0, BATCH_SIZE)
encoder = Encoder(batch_size=BATCH_SIZE)
averages, variances = encoder(images)
results = reparametrization_technique(averages, variances)
print(results.shape)