import os
import argparse
import json
import shutil


def get_config(phase):
	config = Config(phase)
	return config


class Config(object):
	"""Base class of Config, provide necessary hyperparameters. 
	"""
	def __init__(self, phase):
		self.is_train = phase == "train"

		# init hyperparameters and parse from command-line
		parser, args = self.parse()

		# set as attributes
		print("----Experiment Configuration-----")
		for k, v in args.__dict__.items():
			print("{0:20}".format(k), v)
			self.__setattr__(k, v)

		
		# GPU usage
		if args.gpu_ids is not None:
			os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

		
	def parse(self):
		"""initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
		parser = argparse.ArgumentParser()
		
		# basic configuration
		self._add_basic_config_(parser)

		# dataset configuration
		self._add_dataset_config_(parser)

		# model configuration
		self._add_network_config_(parser)

		# training configuration
		self._add_training_config_(parser)

		if not self.is_train:
			# testing configuration
			self._add_testing_config_(parser)

		# additional parameters if needed
		pass

		args = parser.parse_args()
		return parser, args

	def _add_basic_config_(self, parser):
		"""add general hyperparameters"""
		group = parser.add_argument_group('basic')
		group.add_argument('-g', '--gpu_ids', type=str, default=None,
						   help="gpu to use, e.g. 0  0,1,2. CPU not supported.")
		
	def _add_dataset_config_(self, parser):
		"""add hyperparameters for dataset configuration"""
		group = parser.add_argument_group('dataset')
		group.add_argument('--dataset_name', type=str, choices=['partnet', 'partnet_scan', '3depn'], required=True,
						   help="which dataset to use")
		group.add_argument('--data_root', type=str, default="", help="path to complete shape data")
		group.add_argument('--data_raw_root', type=str, default="", help="path to partial shape data")
		group.add_argument('--batch_size', type=int, default=50, help="batch size")
		group.add_argument('--num_workers', type=int, default=8, help="number of workers for data loading")

	def _add_network_config_(self, parser):
		"""add hyperparameters for network architecture"""
		group = parser.add_argument_group('network')

		# VAE
		parser.add_argument('--pretrain_vae_path', type=str,
							help="path for pretrained vae model, only needed when training/testing cGAN")

	def _add_training_config_(self, parser):
		"""training configuration"""
		group = parser.add_argument_group('training')
		group.add_argument('--nr_epochs', type=int, default=2000, help="total number of epochs to train")
		group.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
		group.add_argument('--lr_decay', type=float, default=0.9995, help="step size for learning rate decay")
		
	def _add_testing_config_(self, parser):
		group = parser.add_argument_group('testing')
		group.add_argument('--num_sample', type=int, default=10, help="number test samples to use, -1 for all")
		group.add_argument('--num_z', type=int, default=5, help="number of completion outputs per sample")


if __name__ == '__main__':
	pass