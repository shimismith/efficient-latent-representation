import os
import glob
import numpy as np
from config import get_config
from rgbd_vae import VAEnet


def main():
	#get config
	config = get_config('train')

	#create network
	vae_net = VAEnet()

	trainset = RGBDDataset('train', config.data_root, config.data_raw_root, config.category, config.n_pts)
	valset = RGBDDataset('val', config.data_root, config.data_raw_root, config.category, config.n_pts)

	#get dataloaders 
	train_loader = get_dataloader(trainset,'train',config)
	test_loader = get_dataloader(valset,'val',config)


	for ep in range(config.nr_epochs):

		for batch_idx, data in enumerate(train_loader):

			inp = data['input']
			target = data['target']

			pred = vae_net(inp)

			



if __name__=='__main__':
	main()