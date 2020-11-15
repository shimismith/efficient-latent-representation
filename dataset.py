import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

def get_dataloader(dataset, phase, config):
	is_shuffle = phase == 'train'
		
	dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
							worker_init_fn=np.random.seed())
	return dataloader


class RGBDDataset(Dataset):
	def __init__(self, phase, config):
		if phase == 'train':
			file_list = config.train_file
		elif phase == 'val':
			file_list = config.val_file

		f = open(file_list)
		data_list = f.readlines()
		
		return

	def __getitem__(self,index):
		
		image = Image.open(data_list[index])

		return {'inp':image,'target':image} 


