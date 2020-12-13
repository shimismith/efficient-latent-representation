import numpy as np
from scipy.io import loadmat

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms


class NYU_DepthDataset(Dataset):
    def __init__(self, mat_file, transform=None):
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

      if transform:
        self.rgbd = transform(self.rgbd)

    def __len__(self):
        return self.rgbd.shape[0]

    def __getitem__(self, idx):
      return self.rgbd[idx]


def setup_data_loaders(batch_size, normalize=True):
  nyu = NYU_DepthDataset('/gruvi/usr/shimi/nyu128.mat', transform=transforms.Normalize((0.5,0.5,0.5,0.5), (0.5,0.5,0.5,0.5)))
  split = loadmat('/gruvi/usr/shimi/splits.mat')

  nyu_train = Subset(nyu, split['trainNdxs'].flatten()-1)
  nyu_test = Subset(nyu, split['testNdxs'].flatten()-1)

  train_dloader = DataLoader(dataset=nyu_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
  test_dloader = DataLoader(dataset=nyu_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available())

  return train_dloader, test_dloader
