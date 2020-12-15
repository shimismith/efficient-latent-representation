import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from matplotlib import pyplot as plt

# to plot image reconstructions
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

# device settings
torch.manual_seed(1)
torch.cuda.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


batch_size = 256
# loading MNIST dataset
kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


"""
### To setup NYU V2 dataset ###
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

from google.colab import drive
drive.mount('/content/drive')
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
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


# MSE Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(x_hat, x, mu, logvar):
    mse = nn.functional.mse_loss(x_hat, x.view(-1, 784), reduction='sum')
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
    return mse + KLD


# Training and testing the VAE
epochs = 20
codes = dict(mu1=list(), var1=list(), y=list())
for epoch in range(0, epochs + 1):
    # Training
    if epoch > 0:  # test untrained net first
        model.train()
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss = loss_function(x_hat, x, mu, logvar)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()      
        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
    
    # Testing
    means, logvars, labels = list(), list(), list()
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for x, y in test_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            test_loss += loss_function(x_hat, x, mu, logvar).item()
            means.append(mu.detach())
            logvars.append(logvar.detach())
            labels.append(y.detach())
    codes['mu1'].append(torch.cat(means))
    codes['var1'].append(torch.cat(logvars))
    codes['y'].append(torch.cat(labels))
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    display_images(x, x_hat, 1, f'Epoch {epoch}')
