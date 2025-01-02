# coding:utf8
"""
code refer to https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/transformer_net.py
"""
import torch as t
from torch import nn
import numpy as np


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()

        # Down sample layers
        self.initial_layers = nn.Sequential(
            # Input: [batch_size, 3, H, W]
            ConvLayer(3, 32, kernel_size=9, stride=1),
            # Output: [batch_size, 32, H, W]
            nn.InstanceNorm2d(32, affine=True, track_running_stats=True),
            nn.ReLU(True),
            ConvLayer(32, 64, kernel_size=3, stride=2),
            # Output: [batch_size, 64, H/2, W/2]
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(True),
            ConvLayer(64, 128, kernel_size=3, stride=2),
            # Output: [batch_size, 128, H/4, W/4]
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU(True),
        )

        # Residual layers
        self.res_layers = nn.Sequential(
            # Input: [batch_size, 128, H/4, W/4]
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            # Output: [batch_size, 128, H/4, W/4] (size unchanged)
        )

        # Upsampling Layers
        self.upsample_layers = nn.Sequential(
            # Input: [batch_size, 128, H/4, W/4]
            UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2),
            # Output: [batch_size, 64, H/2, W/2]
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(True),
            UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2),
            # Output: [batch_size, 32, H, W]
            nn.InstanceNorm2d(32, affine=True, track_running_stats=True),
            nn.ReLU(True),
            ConvLayer(32, 3, kernel_size=9, stride=1),
            # Final Output: [batch_size, 3, H, W]
        )

    def forward(self, x):
        # Input: [batch_size, 3, H, W]
        x = self.initial_layers(x)
        # After initial_layers: [batch_size, 128, H/4, W/4]
        x = self.res_layers(x)
        # After res_layers: [batch_size, 128, H/4, W/4]
        x = self.upsample_layers(x)
        # Final output: [batch_size, 3, H, W]
        return x


class ConvLayer(nn.Module):
    """
    add ReflectionPad for Conv
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        # Calculate padding size as half the kernel size
        reflection_padding = int(np.floor(kernel_size / 2))

        # Create reflection padding layer
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)

        # Create convolution layer
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    instead of ConvTranspose2d, we do UpSample + Conv2d
    see ref for why.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        """Forward pass of the convolutional layer with optional upsampling
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Output tensor after padding and convolution
        """
        # Store input tensor for potential upsampling
        x_in = x

        # Optional upsampling step
        if self.upsample:
            # Increase spatial dimensions by self.upsample factor
            # For example, if self.upsample=2:
            # [batch, channels, height, width] -> [batch, channels, height*2, width*2]
            x_in = t.nn.functional.interpolate(x_in, scale_factor=self.upsample)

        # Apply reflection padding to handle border pixels
        # Reflection padding mirrors the border pixels instead of using zeros
        # This helps prevent border artifacts in the output
        out = self.reflection_pad(x_in)

        # Apply convolution operation
        # Transforms the input features according to learned filters
        # Shape changes based on conv2d parameters (kernel_size, stride, etc.)
        out = self.conv2d(out)
        
        return out


class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True, track_running_stats=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out
