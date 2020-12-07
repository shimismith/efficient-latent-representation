import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
from nyu_dataloader import setup_data_loaders

# Input is 216*216
class Encoder(nn.Module):
  def __init__(self, x_dim, z_dim, filters=32):
    super().__init__()

    depths = [x_dim, filters, filters*2, filters*4, filters*8, filters*8]
    convs = []
    for i in range(0, len(depths)-1):
      convs.append(nn.Sequential(nn.Conv2d(depths[i], depths[i+1], 4, padding=1, stride=2), nn.BatchNorm2d(depths[i+1]), nn.LeakyReLU(0.2)))

    self.convs = nn.Sequential(*convs)

    self.fc1 = nn.Linear(depths[-1]*4**2, z_dim * 2)
    # self.fc2 = nn.Linear(depths[-1]*4**2, z_dim)


  def forward(self, x):
    conv_out = self.convs(x).flatten(1)
    x = self.fc1(conv_out)
    # mu = self.fc1(conv_out)
    # log_std = self.fc2(conv_out)
    mean, std = torch.chunk(x, 2, dim=0)
    std = F.softplus(std) + 1e-5
    return Normal(loc=mean, scale=std)


class Decoder(nn.Module):
  def __init__(self, x_dim, z_dim, filters=32, std=1):
    super().__init__()

    depths = [filters*8, filters*8, filters*4, filters*2, filters, x_dim]

    self.fc = nn.Linear(z_dim, depths[0]*4**2)
    self.std = std
    convs = []
    for i in range(0, len(depths)-2):
      convs.append(nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(depths[i], depths[i+1], 3, padding=1, padding_mode='replicate'), nn.BatchNorm2d(depths[i+1]), nn.LeakyReLU(0.2)))

    convs.append(nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(depths[-2], depths[-1], 3, padding=1, padding_mode='replicate')))

    self.convs = nn.Sequential(*convs)

  def forward(self, z):
    fc_out = self.fc(z).view(z.shape[0], -1, 4, 4)
    conv_out = self.convs(fc_out)
    final = torch.tanh(conv_out)

    return Normal(loc=final, scale=torch.ones_like(final) * self.std)


class VAE(nn.Module):
  def __init__(self, x_dim, z_dim, encoder_filters=32, decoder_filters=32):
    super().__init__()
    self.x_dim = x_dim
    self.z_dim = z_dim
    self.encoder = Encoder(self.x_dim, self.z_dim)
    self.decoder = Decoder(self.x_dim, self.z_dim)

    if torch.cuda.is_available():
      self.cuda()

  def reparameterize(self, mu, log_std):
    std = torch.exp(log_std)
    eps = torch.randn_like(std)
    return mu + eps*std

  def forward(self, x):
    latent = self.encoder(x)
    # z = self.reparameterize(z_mu, z_log_std)
    z = latent.rsmaple()
    return self.decoder(z), x


def neg_elbo(reconstructed, x, mu, log_std, kl_coef=1):
    log_likelihood_reconstructed = reconstructed.log_prob(x).mean(dim=0).sum()
    # Compute the reconstruction term - log p(x|z)
    # mse = F.mse_loss(reconstructed, x)

    # -KL for gaussian case: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    log_var = 2*log_std
    kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
    # neg_elbo = mse + kl_coef*kl
    elbo = log_likelihood_reconstructed - kl_coef *kl #TODO: Check dimension match
    return -elbo

# Trains for one epoch
def loss(inputs):

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
    total_epoch_loss_train = epoch_loss / len(train_loader.dataset)
    return total_epoch_loss_train


def evaluate(vae, test_loader):
    vae.eval()
    test_loss = 0
    with torch.no_grad():
      # compute the loss over the entire test set
      for x in test_loader:
        if torch.cuda.is_available():
          x = x.cuda()

        output = vae(x)
        test_loss += neg_elbo(*output, kl_coef=1.0).item()

    total_epoch_loss_test = test_loss / len(test_loader.dataset)
    return total_epoch_loss_test


vae = VAE(4, 400, 64, 64)
optimizer = optim.Adam(vae.parameters(), lr=0.0005)

NUM_EPOCHS = 200
TEST_FREQUENCY = 5
BATCH_SIZE = 50
train_loader, test_loader = setup_data_loaders(batch_size=BATCH_SIZE)

train_elbo = []
test_elbo = []

best = float('inf')

fig, axs = plt.subplots(2, 2)

for epoch in range(NUM_EPOCHS):
    total_epoch_loss_train = train(vae, train_loader, optimizer)
    train_elbo.append(-total_epoch_loss_train)
    print("[epoch %d]  average training loss: %.8f" % (epoch, total_epoch_loss_train))

    if epoch % TEST_FREQUENCY == 0:
        total_epoch_loss_test = evaluate(vae, test_loader)
        test_elbo.append(-total_epoch_loss_test)
        print("[epoch %d] average test loss: %.8f" % (epoch, total_epoch_loss_test))

        # Save stuff
        if total_epoch_loss_test < best:
          print('SAVING EPOCH', epoch)
          best = total_epoch_loss_test
          torch.save({
            'epoch': epoch,
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'rgbd_checkpoint.save')

        if total_epoch_loss_test < 0:  # numerical instability occured
          print('Negative loss occurred!!!', total_epoch_loss_test)
          break

        i = 0
        axs[0, 0].imshow(test_loader.dataset[i][:3].permute(1, 2, 0))
        axs[0, 1].imshow(test_loader.dataset[i][3])

        test_input = test_loader.dataset[i].unsqueeze(0).cuda()
        reconstructed = vae(test_input)[0].cpu().detach()[0]
        axs[1, 0].imshow(reconstructed[:3].permute(1, 2, 0))
        axs[1, 1].imshow(reconstructed[3])
        plt.savefig('rgbd_test.png')
        plt.cla()
