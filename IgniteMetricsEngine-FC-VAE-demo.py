# -*- coding: utf-8 -*-
"""IgniteMetrics-FC.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19cuCPnMuYXdiNoGhTpdVk4MPUCskSy4G
"""

# Commented out IPython magic to ensure Python compatibility.
!pip install pytorch-ignite torchvision


import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
SEED = 1234

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from torchvision.transforms import Compose, ToTensor

from ignite.engine import Engine, Events
from ignite.metrics import MeanSquaredError, Loss, RunningAverage

data_transform = Compose([ToTensor()])

## loading MNIST data ##
train_data = MNIST(download=True, root="/tmp/mnist/", transform=data_transform, train=True)
val_data = MNIST(download=True, root="/tmp/mnist/", transform=data_transform, train=False)

image = train_data[0][0]
label = train_data[0][1]

print ('len(train_data) : ', len(train_data))
print ('len(val_data) : ', len(val_data))
print ('image.shape : ', image.shape)
print ('label : ', label)


kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, **kwargs)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True, **kwargs)

for batch in train_loader:
    x, y = batch
    break

print ('x.shape : ', x.shape)
print ('y.shape : ', y.shape)
fixed_images = x.to(device)

"""
#### To setup NYU V2 dataset ###
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
  train_loader = DataLoader(dataset=nyu_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
  test_loader = DataLoader(dataset=nyu_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available())
  return train_loader, test_loader
"""

d = 10
# define fully connected VAE model
class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, d * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(d, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, 784),
            nn.Sigmoid(),
        )

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu_logvar = self.encoder(x.view(-1, 784)).view(-1, 2, d)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def kld_loss(x_pred, x, mu, logvar):
    mse = nn.functional.mse_loss(x_pred, x.view(-1, 784), reduction='sum')
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
    return mse + KLD

bce_loss = nn.BCELoss(reduction='sum')

def display_images(in_, out, n=1, label=None, count=False):
    for N in range(n):
        if in_ is not None:
            in_pic = in_.data.cpu().view(-1, 28, 28)
            plt.figure(figsize=(18, 4))
            plt.suptitle(label + ' – real test data / reconstructions', color='w', fontsize=16)
            for i in range(4):
                plt.subplot(1,4,i+1)
                plt.imshow(in_pic[i+4*N])
                plt.axis('off')
        out_pic = out.data.cpu().view(-1, 28, 28)
        plt.figure(figsize=(18, 6))
        for i in range(4):
            plt.subplot(1,4,i+1)
            plt.imshow(out_pic[i+4*N])
            plt.axis('off')
            if count: plt.title(str(4 * N + i), color='w')

def train1(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, _ = batch
    x = x.to(device)
    x = x.view(-1, 784)
    x_pred, mu, logvar = model(x)
    MSE = mse_loss(x_pred, x)
    KLD = kld_loss(x_pred, x, mu, logvar)
    loss = MSE + KLD
    loss.backward()
    optimizer.step()
    return loss.item(), BCE.item(), KLD.item()


def test1(engine, batch):
    model.eval()
    with torch.no_grad():
        x, _ = batch
        x = x.to(device)
        x = x.view(-1, 784)
        x_pred, mu, logvar = model(x)
        kwargs = {'mu': mu, 'logvar': logvar}
        display_images(x, x_pred, 1, f'Epoch {batch}')
        return x_pred, x, kwargs

trainer = Engine(train1)
evaluator = Engine(test1)
training_history = {'bce': [], 'kld': [], 'mse': []}
validation_history = {'bce': [], 'kld': [], 'mse': []}

RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'bce')
RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'kld')

MeanSquaredError(output_transform=lambda x: [x[0], x[1]]).attach(evaluator, 'mse')
Loss(bce_loss, output_transform=lambda x: [x[0], x[1]]).attach(evaluator, 'bce')
Loss(kld_loss).attach(evaluator, 'kld')

@trainer.on(Events.EPOCH_COMPLETED)
def print_trainer_logs(engine):
    avg_loss = engine.state.metrics['loss']
    avg_bce = engine.state.metrics['bce']
    avg_kld = engine.state.metrics['kld']
    print("Trainer Results - Epoch {} - Avg loss: {:.2f} Avg bce: {:.2f} Avg kld: {:.2f}"
          .format(engine.state.epoch, avg_loss, avg_bce, avg_kld))
    
def print_logs(engine, dataloader, mode, history_dict):
    evaluator.run(dataloader, max_epochs=1)
    metrics = evaluator.state.metrics
    avg_mse = metrics['mse']
    avg_bce = metrics['bce']
    avg_kld = metrics['kld']
    avg_loss =  avg_mse + avg_kld
    print(
        mode + " Results - Epoch {} - Avg mse: {:.2f} Avg loss: {:.2f} Avg bce: {:.2f} Avg kld: {:.2f}"
        .format(engine.state.epoch, avg_mse, avg_loss, avg_bce, avg_kld))
    for key in evaluator.state.metrics.keys():
        history_dict[key].append(evaluator.state.metrics[key])

trainer.add_event_handler(Events.EPOCH_COMPLETED, print_logs, train_loader, 'Training', training_history)
trainer.add_event_handler(Events.EPOCH_COMPLETED, print_logs, val_loader, 'Validation', validation_history)

e = trainer.run(train_loader, max_epochs=20) ## 20 epochs

plt.plot(range(20), training_history['bce'], 'dodgerblue', label='training')
plt.plot(range(20), validation_history['bce'], 'orange', label='validation')
plt.xlim(0, 20);
plt.xlabel('Epoch')
plt.ylabel('BCE')
plt.title('Binary Cross Entropy on Training/Validation Set')
plt.legend();

plt.plot(range(20), training_history['kld'], 'dodgerblue', label='training')
plt.plot(range(20), validation_history['kld'], 'orange', label='validation')
plt.xlim(0, 20);
plt.xlabel('Epoch')
plt.ylabel('KLD')
plt.title('KL Divergence on Training/Validation Set')
plt.legend();

plt.plot(range(20), training_history['mse'], 'dodgerblue', label='training')
plt.plot(range(20), validation_history['mse'], 'orange', label='validation')
plt.xlim(0, 20);
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Mean Squared Error on Training/Validation Set')
plt.legend();