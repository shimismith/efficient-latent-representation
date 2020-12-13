import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
from nyu_dataloader import setup_data_loaders

from nyu_dataloader_mat import setup_data_loaders
# from nyu_dataloader import setup_data_loaders
from torch.utils.tensorboard import SummaryWriter

# torch.cuda.set_device(1)

# Input is 216*216
class Encoder(nn.Module):
  def __init__(self, x_dim, z_dim, filters=64):
    super().__init__()

    depths = [x_dim, filters, filters*2, filters*4, filters*8, filters*8]
    convs = []
    for i in range(0, len(depths)-1):
        convs.append(nn.Sequential(nn.Conv2d(depths[i], depths[i+1], 4, padding=1, stride=2), nn.BatchNorm2d(depths[i+1]), nn.LeakyReLU(0.2)))

    self.convs = nn.Sequential(*convs)

    self.fc1 = nn.Linear(depths[-1]*4**2, z_dim * 2)

  def forward(self, x):
    conv_out = self.convs(x).flatten(1)
    x = self.fc1(conv_out)
    mean, std = torch.chunk(x, 2, dim=1)
    std = F.softplus(std) + 1e-5
    return Normal(loc=mean, scale=std)


class Decoder(nn.Module):
  def __init__(self, x_dim, z_dim, filters=32):
    super().__init__()

    depths = [filters*8, filters*8, filters*4, filters*2, filters, x_dim]

    self.fc = nn.Linear(z_dim, depths[0]*4**2)

    convs = []
    for i in range(0, len(depths)-2):
      convs.append(nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(depths[i], depths[i+1], 3, padding=1, padding_mode='replicate'), nn.BatchNorm2d(depths[i+1]), nn.LeakyReLU(0.2)))

    convs.append(nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(depths[-2], depths[-1], 3, padding=1, padding_mode='replicate')))

    self.convs = nn.Sequential(*convs)

  def forward(self, z):
    fc_out = self.fc(z).view(z.shape[0], -1, 4, 4)
    conv_out = self.convs(fc_out)
    final = torch.tanh(conv_out)

    return final


class VAE(nn.Module):
  def __init__(self, x_dim, z_dim, encoder_filters=32, decoder_filters=32):
    super().__init__()
    self.x_dim = x_dim
    self.z_dim = z_dim
    self.encoder = Encoder(self.x_dim, self.z_dim)
    self.decoder = Decoder(self.x_dim, self.z_dim)

    if torch.cuda.is_available():
      self.cuda()

  def forward(self, x):
    latent = self.encoder(x)
    z = latent.rsample()
    return self.decoder(z), x, latent

  def reconstruct(self, x):
      latent = self.encoder(x)
      z = latent.rsample()
      return self.decoder(z).mean


def neg_elbo(reconstructed, x, latent):
    log_likelihood_reconstructed = reconstructed.log_prob(x).mean(dim=0).sum()

    # -KL for gaussian case: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # kl = kl_divergence(latent, Normal(torch.zeros_like(latent.loc), torch.ones_like(latent.scale))).sum(1).mean()
    kl = torch.mean(-0.5 * torch.sum(1 + torch.log(latent.variance) - latent.mean.pow(2) - latent.variance, dim=1))

    elbo = log_likelihood_reconstructed - kl
    return -elbo


# Trains for one epoch
def train(vae, train_loader, optimizer):
    vae.train()
    epoch_loss = 0
    for x in train_loader:
      if torch.cuda.is_available():
        x = x.cuda()

      optimizer.zero_grad()
      output = vae(x)
      loss = neg_elbo(*output)
      epoch_loss += loss.item()

      loss.backward()
      optimizer.step()

    # return epoch loss
    total_epoch_loss_train = epoch_loss / len(train_loader)
    return total_epoch_loss_train


def mse(vae, test_loader):
    vae.eval()
    mse = 0
    with torch.no_grad():
      # compute the loss over the entire test set
      for x in test_loader:
        if torch.cuda.is_available():
          x = x.cuda()

        mean = vae.reconstruct(x)
        mse += F.mse_loss(mean, x).item()

    mse = mse / len(test_loader)
    return mse


writer = SummaryWriter(log_dir='/gruvi/usr/shimi/logs/rgbd')

vae = VAE(4, 400, 64, 64)
optimizer = optim.Adam(vae.parameters(), lr=1e-4)

NUM_EPOCHS = 300
TEST_FREQUENCY = 5
BATCH_SIZE = 50
train_loader, test_loader = setup_data_loaders(batch_size=BATCH_SIZE, normalize=True)


best = float('inf')

fig, axs = plt.subplots(2, 2)

for epoch in range(1, NUM_EPOCHS+1):
    total_epoch_loss_train = train(vae, train_loader, optimizer)
    writer.add_scalar('Loss/train', -total_epoch_loss_train, epoch)
    writer.add_scalar('Loss/mse', mse(vae, test_loader), epoch)

    print("[epoch %d]  average training loss: %.8f" % (epoch, total_epoch_loss_train))

    if epoch % TEST_FREQUENCY == 0:
        for i in range(0, 100, 10):
            axs[0, 0].imshow(test_loader.dataset[i][:3].permute(1, 2, 0)*0.5+0.5)
            axs[0, 1].imshow(test_loader.dataset[i][3]*0.5+0.5)
            test_input = test_loader.dataset[i].unsqueeze(0).cuda()
            reconstructed = vae.reconstruct(test_input).cpu().detach()[0]
            axs[1, 0].imshow(reconstructed[:3].permute(1, 2, 0)*0.5+0.5)
            axs[1, 1].imshow(reconstructed[3]*0.5+0.5)
            writer.add_figure('reconstruction{}'.format(i), fig, epoch)
            plt.cla()
