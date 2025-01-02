# coding:utf8
import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        
        # Full VGG16 Structure (we only use first 23 layers, before first FC layer):
        # Block 1 (Input: 3 channels -> 64 channels)
        # Input size: [224x224x3]
        # [0] Conv2d(3, 64, 3x3)    -> [224x224x64]
        # [1] ReLU                   -> [224x224x64]
        # [2] Conv2d(64, 64, 3x3)    -> [224x224x64]
        # [3] ReLU  <- relu1_2       -> [224x224x64]
        # [4] MaxPool2d              -> [112x112x64]
        
        # Block 2 (64 channels -> 128 channels)
        # [5] Conv2d(64, 128, 3x3)   -> [112x112x128]
        # [6] ReLU                   -> [112x112x128]
        # [7] Conv2d(128, 128, 3x3)  -> [112x112x128]
        # [8] ReLU  <- relu2_2       -> [112x112x128]
        # [9] MaxPool2d              -> [56x56x128]
        
        # Block 3 (128 channels -> 256 channels)
        # [10] Conv2d(128, 256, 3x3) -> [56x56x256]
        # [11] ReLU                  -> [56x56x256]
        # [12] Conv2d(256, 256, 3x3) -> [56x56x256]
        # [13] ReLU                  -> [56x56x256]
        # [14] Conv2d(256, 256, 3x3) -> [56x56x256]
        # [15] ReLU <- relu3_3       -> [56x56x256]
        # [16] MaxPool2d             -> [28x28x256]
        
        # Block 4 (256 channels -> 512 channels)
        # [17] Conv2d(256, 512, 3x3) -> [28x28x512]
        # [18] ReLU                  -> [28x28x512]
        # [19] Conv2d(512, 512, 3x3) -> [28x28x512]
        # [20] ReLU                  -> [28x28x512]
        # [21] Conv2d(512, 512, 3x3) -> [28x28x512]
        # [22] ReLU <- relu4_3       -> [28x28x512]
        
        # [23+] Rest of VGG16 (not used):
        # - MaxPool2d                -> [14x14x512]
        # - Block 5: 3x Conv+ReLU    -> [14x14x512]
        # - MaxPool2d                -> [7x7x512]
        # - Fully Connected Layers:
        #   - Flatten               -> [25088] (7*7*512)
        #   - FC(25088, 4096)      -> [4096]
        #   - FC(4096, 4096)       -> [4096]
        #   - FC(4096, 1000)       -> [1000]
        
        # Load pretrained VGG16 and get first 23 layers
        features = list(vgg16(pretrained=True).features)[:23]
        
        # Store features in ModuleList and set to eval mode
        # We don't train VGG16 - just use it as feature extractor
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            # Pass input through each layer
            x = model(x)
            # Collect outputs from specific ReLU layers:
            # relu1_2: Low-level features (edges, colors)
            # relu2_2: Medium-level features (textures)
            # relu3_3: Higher-level features (patterns)
            # relu4_3: High-level features (content)
            if ii in {3, 8, 15, 22}:
                results.append(x)

        # Create named tuple for easy access to specific layer outputs
        # This is better than accessing by index (results[0], results[1], etc.)
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return vgg_outputs(*results)
