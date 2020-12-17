# Efficient Latent Representation Learning

The aim of this project is to develop a VAE model that learns depth information in addition to the RGB data and successfully reconstructs RGB-D 
images. We start by using RGB-D inputs to train a model to reconstruct RGB-D outputs. A number of models were implemented to reconstruct RGB-D images,
and these models have been compared to find the most optimal architecture for RGB-D reconstruction.  

DATSETS USED :  
1. MNIST Dataset of handwritten digits (labeled dataset) – 60,000 training samples and 10,000 test samples of RGB images  
2. NYU Depth Dataset V2 – consists of a variety of indoor scenes recorded by both RGB and Depth cameras with 464 scenes, 407,024 unlabeled frames and
1449 processed pairs of RGB and depth images  

MODELS IMPLEMENTED :  
1. Fully Connected VAE -> "FullyConnected-VAE.py"
2. Conditional VAE -> "Conditional-VAE.py"
3. VGG blocks based VAE -> "VGGblock-VAE.py"  

Link to "nyu.mat" input data file used in the models : https://drive.google.com/file/d/1Fc6PXsGBoG4e6OPTzRbX-Ao826MWS75h/view

PROPOSED VAE MODEL :  
File -> "rgbd_pvae_train.py"  
Description : The encoder consists of 5 convolutional layers each with a kernel size of 4 × 4, 1 pixel of zero-padding and a
stride of 2. Each layer is followed by batch normalization and a leaky relu activation with a slope
of 0.2. The depth in each of these layers is as follows: 64, 128, 256, 512, 512. The convolutional
layers are then followed by a fully connected layer to compute the mean and variance of the latent
variable. In our experiments we use a latent dimension of 400. For the decoder we begin with a fully
connected layer that maps the latent dimension to a 4 × 4 tensor input to the convolutions. We found
that upsampling using nearest neighbour interpolation followed by a convolution produced visually
better results than transpose convolutions. The nearest neighbour interpolation layers are followed by
3 × 3 convolution. All but the last of these layers have batch normalization and leaky relu activations. The depths are decreased
across 5 such layers symmetrically to the encoder. Afterwards we apply a tanh activation to map the
predicted mean to the [−1, 1] range. Our inputs are also normalized into this range.

The weights from our proposed model can be downloaded from https://drive.google.com/file/d/1tfG3A1Ru3uS4IbjeLY2yKJchnudm-9Id/view?usp=sharing

rgbd_pvae_train.py contains the model definition. It can be imported from there and used elsewhere. The file also contains training code, once the dataset is downloaded you can set the paths to where you have it. Additionally you must set a log directory for tensorboard. All results during training are logged to tensorboard for viewing.

DEPTH ESTIMATION FROM RGB IMAGES:
Directory -> depth_estimation

Execution:
1. $ cd depth_estimation

2. Download 'test.zip' and 'models.zip' from Google drive.
Google Drive link: https://drive.google.com/drive/folders/1OuKPWn7w8Fbh--N03IMHAD-KWfy3P5Z_?usp=sharing

3. Unzip 'test.zip' and 'models.zip' in depth_estimation folder.

4. $ python depth_estimation.py arguments_test_nyu.txt
   Note: Please ensure 'rgbd_pvae_train.py' is available in the parent directory.

Results:

RGB-D Reconstructions from RGB images: VAE_results/
Depth Estimations from BTS model: result_bts_nyu_v2_pytorch_densenet161/raw/

Note: For faster execution, please execute on a subset of the dataset by removing some test images from nyudepthv2_test_files_with_gt.txt 
