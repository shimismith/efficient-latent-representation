import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt


bs = 100
# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

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

# initialize random seeds; select gpu device if available
torch.manual_seed(1)
torch.cuda.manual_seed(1) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# function to display the reconstructed images
# this function can be called during training/testing
def display_images(in_, out, n=1, label=None, count=False):
    for N in range(n):
        if in_ is not None:
            in_pic = in_.data.cpu().view(-1, 28, 28)
            plt.figure(figsize=(20, 4))
            plt.suptitle(label + ' – reconstructed images', color='w', fontsize=20)
            for i in range(4):
                plt.subplot(1,4,i+1)
                plt.imshow(in_pic[i+4*N])
                plt.axis('off')
        out_pic = out.data.cpu().view(-1, 28, 28)
        plt.figure(figsize=(20, 6))
        for i in range(4):  
            plt.subplot(1,4,i+1)
            plt.imshow(out_pic[i+4*N])
            plt.axis('off')
            if count: plt.title(str(4 * N + i), color='w')

# Defining Conditional VAE Model
class CVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, c_dim):
        super(CVAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim + c_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim + c_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
    
    def encoder(self, x, c):
        concat_input = torch.cat([x, c], 1)
        h = F.relu(self.fc1(concat_input))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) # return z sample
    
    def decoder(self, z, c):
        concat_input = torch.cat([z, c], 1)
        h = F.relu(self.fc4(concat_input))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))
    
    def forward(self, x, c):
        mu, log_var = self.encoder(x.view(-1, 784), c)
        z = self.sampling(mu, log_var)
        return self.decoder(z, c), mu, log_var

# build model
cond_dim = train_loader.dataset.train_labels.unique().size(0)
cvae = CVAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2, c_dim=cond_dim)
#if torch.cuda.is_available():
#   cvae.cuda()

# print cvae architecture
print(cvae)

optimizer = optim.Adam(cvae.parameters())
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    MSE = F.mse_loss(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE + KLD

# one-hot encoding
def one_hot(labels, class_size): 
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return Variable(targets)

# defining train function
train_loss = []
train_loss.append(0)
def train(epoch):
    cvae.train()
    #train_loss = 0
    for batch_idx, (data, cond) in enumerate(train_loader):
        data, cond = data.to(device), one_hot(cond, cond_dim).to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = cvae(data, cond)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss[-1] += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss[-1] / len(train_loader.dataset)))

# defining test function
def test():
    cvae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, cond in test_loader:
            data, cond = data.to(device), one_hot(cond, cond_dim).to(device)
            recon, mu, log_var = cvae(data, cond)
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    display_images(data, recon, 1, f'Epoch {epoch}')

# training the model over 20 epochs
for epoch in range(1, 20):
    train(epoch)
    test()
