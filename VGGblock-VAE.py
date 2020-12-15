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
pyro.set_rng_seed(100)

"""
### to setup MNIST ###
batch_size = 256

kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
"""

class VGGBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
      super().__init__()
      self.layers=[]
      for _ in range(16):  # num_convs=16 (VGG-16 model) .... To check -> VGG-19
          self.layers.append(nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1))
          self.layers.append(nn.ReLU())
          in_channels = out_channels
      self.layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
      self.linear = None
      if in_channels != out_channels:
          self.linear = nn.Sequential(*self.layers)
      #return nn.Sequential(*layers)

  def forward(self, x):
    if self.linear:
      x = self.linear(x)
    return tf.softmax(x)


# Input is 216*216
class Encoder(nn.Module):
  def __init__(self, z_dim):
    super().__init__()
    
    depths = [64, 76, 88, 100, 128, 200]
    self.conv1 = nn.Sequential(nn.Conv2d(4, 64, 3, padding=1), nn.BatchNorm2d(64), nn.Softplus())

    convs = []
    for i in range(0, len(depths)-1):
      convs.append(nn.Sequential(nn.Conv2d(depths[i], depths[i+1], 3, padding=1, stride=2), nn.BatchNorm2d(depths[i+1]), nn.Softplus()))

    self.convs = nn.Sequential(*convs)
    self.vgg_blocks = nn.Sequential(VGGBlock(depths[-1], depths[-1]), VGGBlock(depths[-1], depths[-1]))

    self.conv2 = nn.Sequential(nn.Conv2d(200, 250, 3, padding=1, stride=2), nn.BatchNorm2d(250), nn.Softplus())
    self.conv3 = nn.Sequential(nn.Conv2d(250, z_dim, 3, padding=1, stride=2), nn.BatchNorm2d(z_dim), nn.Softplus())
    self.fc = nn.Linear(4, 2)

  def forward(self, x):
    conv1_out = self.conv1(x)
    conv_out = self.convs(conv1_out)
    res_out = self.vgg_blocks(conv_out)   
    conv2_out = self.conv2(res_out)
    conv3_out = self.conv3(conv2_out)  
    final = self.fc(conv3_out.flatten(2))  # mu, log(var)
    
    return final


class Decoder(nn.Module):
  def __init__(self, z_dim):
    super().__init__()

    self.conv1 = nn.Sequential(nn.ConvTranspose2d(z_dim, 250, 4), nn.BatchNorm2d(250), nn.Softplus())
    self.conv2 = nn.Sequential(nn.ConvTranspose2d(250, 200, 3), nn.BatchNorm2d(200), nn.Softplus())

    depths = [200, 128, 100, 88, 76, 64]
    self.vgg_blocks = nn.Sequential(VGGBlock(depths[0], depths[0]), VGGBlock(depths[0], depths[0]))

    convs = []
    for i in range(0, 2):
      convs.append(nn.Sequential(nn.ConvTranspose2d(depths[i], depths[i+1], 3, stride=2), nn.BatchNorm2d(depths[i+1]), nn.Softplus()))

    for i in range(2, len(depths)-1):
      convs.append(nn.Sequential(nn.ConvTranspose2d(depths[i], depths[i+1], 2, stride=2), nn.BatchNorm2d(depths[i+1]), nn.Softplus()))

    self.convs = nn.Sequential(*convs)
    self.conv3 = nn.Sequential(nn.Conv2d(64, 4, 3, padding=1), nn.BatchNorm2d(4), nn.Softplus())

  def forward(self, z):
    z = z.view(z.shape[0], z.shape[1], 1, 1)
    conv1_out = self.conv1(z)
    conv2_out = self.conv2(conv1_out)
    res_out = self.vgg_blocks(conv2_out)   ### res_blocks(conv2_out)
    convs_out = self.convs(res_out)
    conv3_out = self.conv3(convs_out)
    final = torch.sigmoid(conv3_out)  # 4 independent bernoulli variables per pixel

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
      params = self.encoder(x)
      z_mu = params[:, :, 0]
      z_log_var = params[:, :, 1]
      pyro.sample("latent", dist.Normal(z_mu, torch.exp(z_log_var)).to_event(1))

  def reconstruct(self, x):
    params = self.encoder(x)
    z_mu = params[:, :, 0]
    z_log_var = params[:, :, 1]
    z = dist.Normal(z_mu, torch.exp(z_log_var)).sample()
    x = self.decoder(z)
    return x


class NYU_DepthDataset(Dataset):
    def __init__(self, mat_file):
      mat = loadmat(mat_file)
      images = torch.from_numpy(mat['images']).permute(3, 2, 0, 1)
      depths = torch.from_numpy(mat['depths']).permute(2, 0, 1)

      images_max = images.amax((2, 3), keepdim=True) 
      images_min = images.amin((2, 3), keepdim=True) 
      images = (images - images_min) / (images_max - images_min)

      depths_max = depths.amax((1, 2), keepdim=True) 
      depths_min = depths.amin((1, 2), keepdim=True) 
      depths = (depths - depths_min) / (depths_max - depths_min)

      self.rgbd = torch.cat((images, depths.unsqueeze(1)), dim=1)

      self.std, self.mean = torch.std_mean(self.rgbd, (0, 2, 3))
      self.rgbd = transforms.functional.normalize(self.rgbd, self.mean, self.std)

    def __len__(self):
        return self.rgbd.shape[0]

    def __getitem__(self, idx):
      return self.rgbd[idx]


def setup_data_loaders(batch_size):
  nyu = NYU_DepthDataset('drive/My Drive/Colab Notebooks/nyu.mat')
  nyu_train = Subset(nyu, range(0, 1159))
  nyu_test = Subset(nyu, range(1159, len(nyu)))

  train_dloader = DataLoader(dataset=nyu_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
  test_dloader = DataLoader(dataset=nyu_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available())

  return train_dloader, test_dloader


# Trains for one epoch
def train(svi, train_loader):
    epoch_loss = 0
    for x in train_loader:
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
    for x in test_loader:
      if torch.cuda.is_available():
          x = x.cuda()

      # compute ELBO estimate and accumulate loss
      test_loss += svi.evaluate_loss(x)

    total_epoch_loss_test = test_loss / len(test_loader.dataset)
    return total_epoch_loss_test


pyro.clear_param_store()
vae = VAE(400)
optimizer = Adam({"lr": 1.0e-5})

# num_particles defaults to 1. Can increase to get ELBO over multiple samples of z~q(z|x).
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

NUM_EPOCHS = 500  
TEST_FREQUENCY = 5
BATCH_SIZE = 50
train_loader, test_loader = setup_data_loaders(batch_size=BATCH_SIZE)

train_elbo = []
test_elbo = []

vae.train()

best = float('inf')

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
        if (total_epoch_loss_test < best):
          print('SAVING EPOCH', epoch)
          best = total_epoch_loss_test
          pyro.get_param_store().save('drive/My Drive/pyro_weights.save')
          optimizer.save('drive/My Drive/optimizer_state.save')
          checkpoint = {'model_state_dict': vae.state_dict()}
          torch.save(checkpoint, 'drive/My Drive/torch_weights.save')

        i = 0
        fig = plt.figure()
        fig.add_subplot(2, 2, 1)
        plt.imshow(test_loader.dataset[i][:3].permute(1, 2, 0)*test_loader.dataset.dataset.std[:3] + test_loader.dataset.dataset.mean[:3])
        fig.add_subplot(2, 2, 2)
        plt.imshow(test_loader.dataset[i][3]*test_loader.dataset.dataset.std[3] + test_loader.dataset.dataset.mean[3])

        test_input = test_loader.dataset[i].unsqueeze(0)#.cuda()
        reconstructed = vae.reconstruct(test_input).cpu().detach()[0]
        fig.add_subplot(2, 2, 3)
        plt.imshow(reconstructed[:3].permute(1, 2, 0)*test_loader.dataset.dataset.std[:3] + test_loader.dataset.dataset.mean[:3])
        fig.add_subplot(2, 2, 4)
        plt.imshow(reconstructed[3]*test_loader.dataset.dataset.std[3] + test_loader.dataset.dataset.mean[3])
        plt.show()
