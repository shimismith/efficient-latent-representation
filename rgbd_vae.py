import torch
import torch.nn as nn
import torch.nn.functional as tf
import numpy as np

# Deep Residual Learning for Image Recognition: https://arxiv.org/pdf/1512.03385.pdf
class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ResBlock, self).__init__()

    self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(),
                              nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels))
    
    self.linear = None
    if in_channels != out_channels:
      self.linear = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, padding=0), nn.BatchNorm2d(out_channels))

  def forward(self, x):
    if self.linear:
      x = self.linear(x)
    return tf.relu(self.conv(x) + x)

# TODO replace batch norm with instance norm maybe?
# Input is 216*216
class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    
    depths = [64, 76, 88, 100, 128, 200]
    self.conv1 = nn.Sequential(nn.Conv2d(4, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU())

    convs = []
    for i in range(0, len(depths)-1):
      convs.append(nn.Sequential(nn.Conv2d(depths[i], depths[i+1], 3, padding=1, stride=2), nn.BatchNorm2d(depths[i+1]), nn.ReLU()))

    self.convs = nn.Sequential(*convs)
    self.res_blocks = nn.Sequential(ResBlock(depths[-1], depths[-1]), ResBlock(depths[-1], depths[-1]))

    self.conv2 = nn.Sequential(nn.Conv2d(200, 250, 3, padding=1, stride=2), nn.BatchNorm2d(250), nn.ReLU())
    self.conv3 = nn.Sequential(nn.Conv2d(250, 300, 3, padding=1, stride=2), nn.BatchNorm2d(300), nn.ReLU())
    self.fc = nn.Linear(4, 2)

  def forward(self, x):
    out = self.conv1(x)
    conv_out = self.convs(out)
    res_out = self.res_blocks(conv_out)
    conv2_out = self.conv2(res_out)
    conv3_out = self.conv3(conv2_out)  # mu, log(var)

    final = conv3_out.flatten(2)
    final = self.fc(final)
    
    return final

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()

    self.conv1 = nn.Sequential(nn.ConvTranspose2d(300, 250, 4), nn.BatchNorm2d(250), nn.ReLU())
    self.conv2 = nn.Sequential(nn.ConvTranspose2d(250, 200, 3), nn.BatchNorm2d(200), nn.ReLU())

    depths = [200, 128, 100, 88, 76, 64]
    self.res_blocks = nn.Sequential(ResBlock(depths[0], depths[0]), ResBlock(depths[0], depths[0]))

    convs = []
    for i in range(0, 2):
      convs.append(nn.Sequential(nn.ConvTranspose2d(depths[i], depths[i+1], 3, stride=2), nn.BatchNorm2d(depths[i+1]), nn.ReLU()))

    for i in range(2, len(depths)-1):
      convs.append(nn.Sequential(nn.ConvTranspose2d(depths[i], depths[i+1], 2, stride=2), nn.BatchNorm2d(depths[i+1]), nn.ReLU()))

    self.convs = nn.Sequential(*convs)
    self.conv3 = nn.Sequential(nn.Conv2d(64, 4, 3, padding=1), nn.BatchNorm2d(4), nn.LeakyReLU())

  def forward(self, x):
    x = x.unsqueeze(-1)
    conv1_out = self.conv1(x)
    conv2_out = self.conv2(conv1_out)
    res_out = self.res_blocks(conv2_out)
    convs_out = self.convs(res_out)
    conv3_out = self.conv3(convs_out)  # 4 independent bernoulli vars

