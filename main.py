import torch
import torch.linalg
import preprocessing as prep
from net import VAE
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
SIZE = 512
LATENT = 128
BATCH_SIZE = 10
EPOCHS = 50  
LEARNING_RATE = 1e-2

def KL_divergence_one_value(avg : torch.Tensor, var : torch.Tensor):
    return torch.trace(torch.exp(var)) + torch.pow(torch.norm(avg), 2) - \
            LATENT - torch.linalg.det(torch.exp(var))

def ELBO(x, x_recon, avg, var):
    BCE = torch.nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
    return BCE - KL_divergence_one_value(avg, var)

model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
X = prep.get_only_training_image(0, BATCH_SIZE)
X = X / 255
X_recon, averages, variances = model(X)
torch.autograd.set_detect_anomaly(True)
for epoch in range(EPOCHS):
    model.train() 
    X = prep.get_only_training_image(0, BATCH_SIZE)
    X = X / 255.0  
    X_recon, averages, variances = model(X)
    for i in range(BATCH_SIZE):
        loss = ELBO(X[i], X_recon[i], averages[i], variances[i])
        print(loss)
        loss.backward(retain_graph=True)  # Backpropagation to compute gradients
        optimizer.zero_grad()  # Reset gradients from the previous step
        optimizer.step()       # Gradient descent update step

    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{EPOCHS}, Loss: {loss.item()}')

print("Training complete.")

