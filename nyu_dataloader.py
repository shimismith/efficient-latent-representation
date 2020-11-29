# Reference: https://github.com/fangchangma/sparse-to-dense.pytorch/blob/master/dataloaders

import os
import h5py
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


def find_classes(root):
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    return classes

def get_image_paths(root, classes):
    images = []
    for scene in classes:
        d = os.path.join(root, scene)
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if fname.endswith('.h5'):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    depth = np.array(h5f['depth'])
    return rgb, depth


class NYU_DepthDataset(Dataset):
    def __init__(self, root, resize=None, normalize=None):
      classes = find_classes(root)
      images = get_image_paths(root, classes)
      self.images = images
      self.resize = resize
      self.normalize = normalize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
      path = self.images[idx]
      rgb, depth = h5_loader(path)

      rgb = torch.from_numpy(rgb)
      depth = torch.from_numpy(depth).unsqueeze(0)

      if self.resize:
          rgb = self.resize(rgb)
          depth = self.resize(depth)

      rgb_max = rgb.amax((1, 2), keepdim=True)
      rgb_min = rgb.amin((1, 2), keepdim=True)
      rgb = (rgb - rgb_min) / (rgb_max - rgb_min)

      depth_max = depth.amax((1, 2), keepdim=True)
      depth_min = depth.amin((1, 2), keepdim=True)
      depth = (depth - depth_min) / (depth_max - depth_min)

      rgbd = torch.cat((rgb, depth))

      if self.normalize:
          rgbd = self.normalize(rgbd)

      return rgbd


def setup_data_loaders(batch_size, normalize=True):
  resize = transforms.Resize((128, 128))
  normalize = None
  if normalize:
      normalize = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))

  nyu_train = NYU_DepthDataset('/gruvi/usr/shimi/nyudepthv2/train', resize=resize, normalize=normalize)
  nyu_test = NYU_DepthDataset('/gruvi/usr/shimi/nyudepthv2/val', resize=resize, normalize=normalize)

  train_dloader = DataLoader(dataset=nyu_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
  test_dloader = DataLoader(dataset=nyu_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available())

  return train_dloader, test_dloader
