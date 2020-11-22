import torch
import torch.nn as nn
import torch.nn.functional as tf
import numpy as np
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from torch.utils.data import DataLoader, Dataset, Subset
from scipy.io import loadmat
import torchvision.transforms as transforms
import torchvision.datasets as datasets


pyro.set_rng_seed(101)

# Input is 216*216
class Encoder(nn.Module):
  def __init__(self, z_dim):
    super().__init__()
    
    depths = [3, 32, 64, 128, 256]
    convs = []
    for i in range(0, len(depths)-1):
      convs.append(nn.Sequential(nn.Conv2d(depths[i], depths[i+1], 4, padding=1, stride=2), nn.BatchNorm2d(depths[i+1]), nn.LeakyReLU()))
    
    self.convs = nn.Sequential(*convs)

    self.fc1 = nn.Linear(4096, z_dim)
    self.fc2 = nn.Linear(4096, z_dim)


  def forward(self, x):
    conv_out = self.convs(x).flatten(1)

    mu = self.fc1(conv_out)
    log_var = self.fc2(conv_out)
    
    return mu, log_var


class Decoder(nn.Module):
  def __init__(self, z_dim):
    super().__init__()

    self.fc = nn.Linear(z_dim, 4096)

    depths = [256, 128, 64, 32, 3]
    convs = []
    for i in range(0, len(depths)-1):
      convs.append(nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(depths[i], depths[i+1], 3, padding=1, padding_mode='replicate'), nn.BatchNorm2d(depths[i+1]), nn.LeakyReLU()))

    self.convs = nn.Sequential(*convs)

  def forward(self, z):
    fc_out = self.fc(z).view(z.shape[0], 256, 4, 4)
    conv_out = self.convs(fc_out)
    final = torch.sigmoid(conv_out)  # 4 independent bernoulli variables per pixel

    return final


class VAE(nn.Module):
  def __init__(self, z_dim):
    super().__init__()
    self.z_dim = z_dim
    self.encoder = Encoder(self.z_dim)
    self.decoder = Decoder(self.z_dim)

    if torch.cuda.is_available():
      self.cuda()
        
  # p(x, z) = p(x|z)p(z)
  def model(self, x):
    pyro.module("decoder", self.decoder)
    with pyro.plate("data", x.shape[0]):
      # mean and variance of prior p(z)
      z_mu = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
      z_var = x.new_ones(torch.Size((x.shape[0], self.z_dim)))

      z = pyro.sample("latent", dist.Normal(z_mu, z_var).to_event(1))
      x_means = self.decoder(z)
      pyro.sample("obs", dist.Bernoulli(x_means).to_event(3), obs=x)

  # approximate posterior q(z|x)
  def guide(self, x):
    pyro.module("encoder", self.encoder)
    with pyro.plate("data", x.shape[0]):
      z_mu, z_log_var = self.encoder(x)
      pyro.sample("latent", dist.Normal(z_mu, torch.exp(z_log_var)).to_event(1))

  def reconstruct(self, x):
    z_mu, z_log_var = self.encoder(x)
    z = dist.Normal(z_mu, torch.exp(z_log_var)).sample()
    x = self.decoder(z)
    return x



def setup_data_loaders(batch_size):
  transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  trainset = datasets.CelebA('./data', split='train', download=True, transform=transform)
  testset = datasets.CelebA('./data', split='test', download=True, transform=transform)

  train_dloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
  test_dloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available())

  return train_dloader, test_dloader


# Trains for one epoch
def train(svi, train_loader):
    epoch_loss = 0
    for x, _ in train_loader:
      if torch.cuda.is_available():
        x = x.cuda()

        # compute ELBO gradient and accumulate loss
        epoch_loss += svi.step(x)

    # return epoch loss
    total_epoch_loss_train = epoch_loss / len(train_loader.dataset)
    return total_epoch_loss_train


def evaluate(svi, test_loader, use_cuda=False):
    test_loss = 0
    # compute the loss over the entire test set
    for x, _ in test_loader:
      if torch.cuda.is_available():
          x = x.cuda()

      # compute ELBO estimate and accumulate loss
      test_loss += svi.evaluate_loss(x)

    total_epoch_loss_test = test_loss / len(test_loader.dataset)
    return total_epoch_loss_test


pyro.clear_param_store()
# pyro.enable_validation(True)
# pyro.distributions.enable_validation(False)

vae = VAE(100)
optimizer = Adam({"lr": 0.0005})

# num_particles defaults to 1. Can increase to get ELBO over multiple samples of z~q(z|x).
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

# pyro.get_param_store().load('pyro_weights.save')
# pyro.module("decoder", vae.decoder, update_module_params=True)
# pyro.module("encoder", vae.encoder, update_module_params=True)
# optimizer.load('optimizer_state.save')

NUM_EPOCHS = 20
TEST_FREQUENCY = 5
BATCH_SIZE = 50
train_loader, test_loader = setup_data_loaders(batch_size=BATCH_SIZE)

train_elbo = []
test_elbo = []

vae.eval()
best = float('inf')

vae.train()

fig, axs = plt.subplots(1, 2)

for epoch in range(NUM_EPOCHS):
    total_epoch_loss_train = train(svi, train_loader)
    train_elbo.append(-total_epoch_loss_train)
    print("[epoch %d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

    if epoch % TEST_FREQUENCY == 0:
        vae.eval()
        total_epoch_loss_test = evaluate(svi, test_loader)
        vae.train()
        test_elbo.append(-total_epoch_loss_test)
        print("[epoch %d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

        # Save stuff
        if total_epoch_loss_test < best:
          print('SAVING EPOCH', epoch)
          best = total_epoch_loss_test
          optimizer.save('celeba_optimizer_state.save')
          checkpoint = {'model_state_dict': vae.state_dict()}
          torch.save(checkpoint, 'celeba_torch_weights.save')

        if total_epoch_loss_test < 0:  # numerical instability occured
          print('Negative loss occurred!!!', total_epoch_loss_test)
          break

        i = 0
        axs[0].imshow(test_loader.dataset[i][0].permute(1, 2, 0))

        test_input = test_loader.dataset[i][0].unsqueeze(0).cuda()
        reconstructed = vae.reconstruct(test_input).cpu().detach()[0]
        axs[1].imshow(reconstructed.permute(1, 2, 0))
        plt.savefig('test.png')
        plt.cla()
