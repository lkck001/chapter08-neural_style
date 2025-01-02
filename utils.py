# coding:utf8
from itertools import chain
import visdom
import torch as t
import time
import torchvision as tv
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def gram_matrix(y):
    """
    Compute the Gram matrix from features to capture style information
    Input shape: b,c,h,w (batch, channels, height, width)
    Output shape: b,c,c (batch, channels, channels)
    """
    # Get dimensions from input tensor
    (b, ch, h, w) = y.size()
    
    # Reshape the features to (batch, channels, height*width)
    # This flattens the spatial dimensions (h,w) into one dimension
    features = y.view(b, ch, w * h)
    
    # Transpose to (batch, height*width, channels)
    # This prepares for matrix multiplication
    features_t = features.transpose(1, 2)
    
    # Compute Gram matrix:
    # - bmm: batch matrix multiplication
    # - Divide by (ch * h * w) to normalize
    # - Result is correlation between feature maps
    gram = features.bmm(features_t) / (ch * h * w)
    
    return gram


class Visualizer:
    """
    wrapper on visdom, but you may still call native visdom by `self.vis.function`
    """

    def __init__(self, env="default", **kwargs):
        import visdom

        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        self.index = {}
        self.log_text = ""

    def reinit(self, env="default", **kwargs):
        """ """
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        return self

    def plot_many(self, d):
        """
        plot multi values in a time
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(
            Y=np.array([y]),
            X=np.array([x]),
            win=name,
            opts=dict(title=name),
            update=None if x == 0 else "append",
        )
        self.index[name] = x + 1

    def img(self, name, img_):
        """
        self.img('input_img',t.Tensor(64,64))
        """

        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.image(img_.cpu(), win=name, opts=dict(title=name))

    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        """
        convert batch images to grid of images
        i.e. input（36，64，64） ->  6*6 grid，each grid is an image of size 64*64
        """
        self.img(
            name, tv.utils.make_grid(input_3d.cpu()[0].unsqueeze(1).clamp(max=1, min=0))
        )

    def log(self, info, win="log_text"):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += "[{time}] {info} <br>".format(
            time=time.strftime("%m%d_%H%M%S"), info=info
        )
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, name):
        return getattr(self.vis, name)


def get_style_data(path):
    """
    load style image，
    Return： tensor shape 1*c*h*w, normalized
    """
    style_transform = tv.transforms.Compose(
        [
            # 1. ToTensor():
            #    - Converts PIL Image or numpy array to PyTorch tensor
            #    - Scales image pixels from 0-255 to 0-1
            #    - Changes shape from (H,W,C) to (C,H,W)
            tv.transforms.ToTensor(),
            # 2. Normalize():
            #    - Normalizes tensor with ImageNet mean and std
            #    - For each channel: output = (input - mean) / std
            #    - Makes model inputs consistent with VGG training data
            #    - IMAGENET_MEAN = [0.485, 0.456, 0.406]
            #    - IMAGENET_STD = [0.229, 0.224, 0.225]
            tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # Load image from file path using PIL (Python Imaging Library)
    # default_loader automatically handles different image formats (PNG, JPEG, etc.)
    style_image = tv.datasets.folder.default_loader(path)

    # Apply our transforms (ToTensor and Normalize) to convert image to normalized tensor
    # Output shape: (C, H, W) - 3 channels, Height, Width
    style_tensor = style_transform(style_image)

    # Add batch dimension: (C,H,W) -> (1,C,H,W)
    # Neural networks expect batched input, even for single images
    # Output shape: (1, C, H, W) - Batch size 1, 3 channels, Height, Width
    return style_tensor.unsqueeze(0)


def normalize_batch(batch):
    """
    Input: b,ch,h,w  0~255    # batch size, channels, height, width
    Output: b,ch,h,w  -2~2    # normalized values typically between -2 and 2
    """
    # Create tensors for ImageNet mean and std with proper shape: (1, channels, 1, 1)
    mean = batch.data.new(IMAGENET_MEAN).view(1, -1, 1, 1)
    std = batch.data.new(IMAGENET_STD).view(1, -1, 1, 1)
    
    # Expand mean and std to match batch dimensions
    mean = mean.expand_as(batch.data)
    std = std.expand_as(batch.data)
    
    # Normalize using two steps:
    # 1. Divide by 255 to scale from [0,255] to [0,1]
    # 2. Apply (value - mean) / std normalization
    return (batch / 255.0 - mean) / std
