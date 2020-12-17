#extended from https://github.com/cogaplex-bts/bts.git
from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.autograd import Variable
from bts_dataloader import *

import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

#from nyu_dataloader_mat import setup_data_loaders
from nyu_dataloader_depth_est import setup_data_loaders
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from rgbd_pvae_train import VAE


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='bts_nyu_v2')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='densenet161_bts')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
    """Test function."""
    args.mode = 'test'
    dataloader = BtsDataLoader(args, 'test')
    #initialize random seeds; select gpu device if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    num_test_samples = get_num_lines(args.filenames_file)
    print(args.filenames_file)

    with open(args.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))
   
    pred_depths = []
    pred_8x8s = []
    pred_4x4s = []
    pred_2x2s = []
    pred_1x1s = []

    start_time = time.time()
    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader.data)):
            image = Variable(sample['image'].to(device))
            focal = Variable(sample['focal'].to(device))
            # Predict
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
            pred_depths.append(depth_est.cpu().numpy().squeeze())
            pred_8x8s.append(lpg8x8[0].cpu().numpy().squeeze())
            pred_4x4s.append(lpg4x4[0].cpu().numpy().squeeze())
            pred_2x2s.append(lpg2x2[0].cpu().numpy().squeeze())
            pred_1x1s.append(reduc1x1[0].cpu().numpy().squeeze())

    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')
    
    save_name = 'result_' + args.model_name
    
    print('Saving result pngs..')
    if not os.path.exists(os.path.dirname(save_name)):
        try:
            os.mkdir(save_name)
            os.mkdir(save_name + '/raw')
            os.mkdir(save_name + '/cmap')
            os.mkdir(save_name + '/rgb')
            os.mkdir(save_name + '/gt')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
    for s in tqdm(range(num_test_samples)):
        scene_name = lines[s].split()[0].split('/')[0]
        filename_pred_png = save_name + '/raw/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
            '.jpg', '.png')
        filename_cmap_png = save_name + '/cmap/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
            '.jpg', '.png')
        filename_gt_png = save_name + '/gt/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
            '.jpg', '.png')
        filename_image_png = save_name + '/rgb/' + scene_name + '_' + lines[s].split()[0].split('/')[1]
    
        rgb_path = os.path.join(args.data_path, './' + lines[s].split()[0])
        image = cv2.imread(rgb_path)
        if args.dataset == 'nyu':
            gt_path = os.path.join(args.data_path, './' + lines[s].split()[1])
            gt = cv2.imread(gt_path, -1).astype(np.float32) / 1000.0  # Visualization purpose only
            gt[gt == 0] = np.amax(gt)
        
        pred_depth = pred_depths[s]
        pred_8x8 = pred_8x8s[s]
        pred_4x4 = pred_4x4s[s]
        pred_2x2 = pred_2x2s[s]
        pred_1x1 = pred_1x1s[s]
        
        if args.dataset == 'kitti' or args.dataset == 'kitti_benchmark':
            pred_depth_scaled = pred_depth * 256.0
        else:
            pred_depth_scaled = pred_depth * 1000.0
        
        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        if args.save_lpg:
            cv2.imwrite(filename_image_png, image[10:-1 - 9, 10:-1 - 9, :])
            if args.dataset == 'nyu':
                plt.imsave(filename_gt_png, np.log10(gt[10:-1 - 9, 10:-1 - 9]), cmap='Greys')
                pred_depth_cropped = pred_depth[10:-1 - 9, 10:-1 - 9]
                plt.imsave(filename_cmap_png, np.log10(pred_depth_cropped), cmap='Greys')
                pred_8x8_cropped = pred_8x8[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8_cropped), cmap='Greys')
                pred_4x4_cropped = pred_4x4[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4_cropped), cmap='Greys')
                pred_2x2_cropped = pred_2x2[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2_cropped), cmap='Greys')
                pred_1x1_cropped = pred_1x1[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_1x1_cropped), cmap='Greys')
            else:
                plt.imsave(filename_cmap_png, np.log10(pred_depth), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_1x1), cmap='Greys')
    
    return num_test_samples

def toVAE(num_test_samples):
    BATCH_SIZE = 50
    test_loader = setup_data_loaders(batch_size=BATCH_SIZE, normalize=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('./models/rgbd_pvae/rgbd_pvae.save', map_location=torch.device('cpu'))
    vae = VAE(4,400,64,64)
    vae.load_state_dict(checkpoint)  
    if not os.path.exists(os.path.dirname('VAE_results')):
        try:
            os.mkdir('VAE_results')
            os.mkdir('VAE_results' + '/pred_rgb')
            os.mkdir('VAE_results' + '/pred_depth')
            os.mkdir('VAE_results' + '/gt_rgb')
            os.mkdir('VAE_results' + '/gt_depth')

        except OSError as e:
            if e.errno != errno.EEXIST:
                raise 
    num_test_samples -=1
    for i in range(0,num_test_samples):
        gt_rgb = test_loader.dataset[i][:3].permute(1, 2, 0)*0.5 + 0.5
        gt_depth = test_loader.dataset[i][3]
        #input = torch.cat((test_loader.dataset[i][:3], test_loader.dataset[i][3])).unsqueeze(0).to(device)

        reconstructed = vae.reconstruct(test_loader.dataset[i].unsqueeze(0).to(device)).cpu().detach()[0]
        reconstruct_rgb = reconstructed[:3].permute(1, 2, 0)*0.5 + 0.5
        reconstruct_depth = reconstructed[3]

        plt.imsave(fname='VAE_results/pred_rgb/reconstruct_rgb'+str(i)+'.png', arr=reconstruct_rgb.numpy(), format='png')
        plt.imsave(fname='VAE_results/pred_depth/reconstruct_depth'+str(i)+'.png', arr=reconstruct_depth.numpy(), format='png')
        plt.imsave(fname='VAE_results/gt_rgb/gt_rgb'+str(i)+'.png', arr=gt_rgb.numpy(), format='png')
        plt.imsave(fname='VAE_results/gt_depth/gt_depth'+str(i)+'.png', arr=gt_depth.numpy(), format='png')
       
if __name__ == '__main__':
    num_test_samples = test(args)
    #num_test_samples = 11
    
    toVAE(num_test_samples)
