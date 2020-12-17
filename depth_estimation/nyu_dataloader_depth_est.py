import numpy as np
import glob
from scipy.io import loadmat
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms


class NYU_DepthDataset(Dataset):
    def __init__(self,  transform=None):
      filenames = glob.glob('./test/*/rgb*.*')
      count = 0
      images =[]
      depths =[]
      
      for f in filenames:
        images.append(np.array(Image.open(f)))
        depths.append(np.array(Image.open(f.replace('./test\\','./result_bts_nyu_v2_pytorch_densenet161\\raw\\').replace('\\rgb','_rgb').replace('.jpg','.png'))))

      images = np.stack(images)
      depths = np.stack(depths)
      print(images.shape)
      print(depths.shape)

      
      images = torch.from_numpy(images).permute(0, 3, 1, 2)
      depths = torch.from_numpy(depths)#.permute(2, 0, 1)
      print('Permuted images:', images.shape)
      print('Permuted depths:', depths.shape)


      images_max = images.amax((2, 3), keepdim=True)
      images_min = images.amin((2, 3), keepdim=True)
      images = (images - images_min) / (images_max - images_min)

      depths_max = depths.amax((1, 2), keepdim=True)
      depths_min = depths.amin((1, 2), keepdim=True)
      depths = (depths - depths_min) / (depths_max - depths_min)

      self.rgbd = torch.cat((images, depths.unsqueeze(1)), dim=1)
      #resize to 128*128
      
      #self.rgbd = transforms.Compose([transforms.Resize((128,128))])

      if transform:
        self.rgbd = transform(self.rgbd)

    def __len__(self):
        return self.rgbd.shape[0]

    def __getitem__(self, idx):
      return self.rgbd[idx]


def setup_data_loaders(batch_size, normalize=True):

  nyu = NYU_DepthDataset(transform=transforms.Compose([transforms.Resize((128,128)), transforms.Normalize((0.5,0.5,0.5,0.5), (0.5,0.5,0.5,0.5))]))
  
  print(nyu)

  test_dloader = DataLoader(dataset=nyu, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available())

  return test_dloader
