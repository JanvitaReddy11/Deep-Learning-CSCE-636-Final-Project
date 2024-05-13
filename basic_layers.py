import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, middle_width_factor=2):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.middle_width_factor = middle_width_factor
        
        # Define the first batch normalization layer
        self.bn1 = nn.BatchNorm2d(input_channels)
        
        # Define the first ReLU activation function
        self.relu1 = nn.ReLU(inplace=True)
        
        # Define the first convolution layer
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride, padding=1, bias=False)
        
        # Define the second batch normalization layer
        self.bn2 = nn.BatchNorm2d(output_channels)
        
        # Define the second ReLU activation function
        self.relu2 = nn.ReLU(inplace=True)
        
        # Define the second convolution layer
        self.conv2 = nn.Conv2d(output_channels, output_channels * self.middle_width_factor, 1, 1, bias=False)
        
        # Define the third batch normalization layer
        self.bn3 = nn.BatchNorm2d(output_channels * self.middle_width_factor)
        
        # Define the third ReLU activation function
        self.relu3 = nn.ReLU(inplace=True)
        
        # Define the third convolution layer
        self.convadd = nn.Conv2d(output_channels * self.middle_width_factor, output_channels* self.middle_width_factor, 3, 1,padding=1, bias=False)
        self.bnadd = nn.BatchNorm2d(output_channels * self.middle_width_factor)
        self.reluadd = nn.ReLU(inplace=True)
        
        # Define the third ReLU activation function
    
        self.conv3 = nn.Conv2d(output_channels * self.middle_width_factor, output_channels, 3, 1,padding=1, bias=False)
        
        
        # If the input and output channels are not the same, apply a convolutional layer to match the dimensions
        if input_channels != output_channels or stride != 1:
            self.shortcut = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)
        else:
            self.shortcut = nn.Identity()  # Identity mapping if input and output channels are the same
        
    def forward(self, x):
        residual = x
        
        # First batch normalization layer
        out = self.bn1(x)
        
        # First ReLU activation function
        out = self.relu1(out)
        
        # First convolution layer
        out = self.conv1(out)
        
        # Second batch normalization layer
        out = self.bn2(out)
        
        # Second ReLU activation function
        out = self.relu2(out)
        
        # Second convolution layer
        out = self.conv2(out)
        
        # Third batch normalization layer
        out = self.bn3(out)
        
        # Third ReLU activation function
        out = self.relu3(out)
        out = self.convadd(out)
        out = self.bnadd(out)
        out = self.reluadd(out)
        # Third convolution layer
        out = self.conv3(out)
        
        # Shortcut connection
        shortcut = self.shortcut(residual)
        
        # Residual connection
        out += shortcut
        
        return out
