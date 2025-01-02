# coding:utf8

import torch as t
import torchvision as tv
import torchnet as tnt
from torch.utils import data
from transformer_net import TransformerNet
import utils
from PackedVGG import Vgg16
from torch.nn import functional as F
import tqdm
import os
import ipdb

# Normalization constants for image preprocessing
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class Config(object):
    """Configuration class that holds all training and testing parameters"""

    # General Args
    use_gpu = False  # Whether to use GPU for computation
    model_path = None  # Path to pretrained model for resume training or testing
    enable_vis = False  # Whether to enable visualization during training

    # Training Args
    image_size = 256  # Size to crop training images to
    batch_size = 8  # Number of images per training batch
    data_root = "data/"  # Root directory containing training images
    num_workers = 4  # Number of parallel workers for data loading

    lr = 1e-3  # Learning rate for optimization
    epoches = 2  # Number of training epochs
    content_weight = 1e5  # Weight factor for content loss
    style_weight = 1e10  # Weight factor for style loss

    style_path = "style.jpg"  # Path to style image used for training
    env = "neural-style"  # Visdom environment name for visualization
    plot_every = 10  # Frequency of visualization updates

    debug_file = "/tmp/debugnn"  # File to trigger debug mode

    # Test Args
    content_path = "input.png"  # Path to content image for style transfer
    result_path = "output.png"  # Path to save style transfer result


def train(**kwargs):
    """Main training function for the style transfer model

    Args:
        **kwargs: Overrides for Config parameters
    """
    # Initialize configuration
    opt = Config()
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    # Check and print device information
    device = (
        t.device("cuda") if opt.use_gpu and t.cuda.is_available() else t.device("cpu")
    )
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU Device: {t.cuda.get_device_name(0)}")
        print(
            f"Available GPU memory: {t.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # Initialize visualization if enabled
    vis = None
    if opt.enable_vis:
        vis = utils.Visualizer(opt.env)

    # Set up data loading pipeline
    transfroms = tv.transforms.Compose(
        [
            tv.transforms.Resize(opt.image_size),
            tv.transforms.CenterCrop(opt.image_size),
            tv.transforms.ToTensor(),
            tv.transforms.Lambda(lambda x: x * 255),
        ]
    )

    dataset = tv.datasets.ImageFolder(opt.data_root, transfroms)
    dataloader = data.DataLoader(dataset, opt.batch_size)

    print("\nDataset information:")
    print(f"Total images: {len(dataset)}")
    print(f"Batch size: {opt.batch_size}")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Total batches for {opt.epoches} epochs: {len(dataloader) * opt.epoches}")

    # Initialize style transfer network
    transformer = TransformerNet()
    if opt.model_path:
        transformer.load_state_dict(
            t.load(opt.model_path, map_location=lambda _s, _: _s)
        )
    transformer.to(device)

    # Initialize VGG network for perceptual loss
    vgg = Vgg16().eval()
    vgg.to(device)

    for param in vgg.parameters():
        param.requires_grad = False

    # Set up optimizer
    optimizer = t.optim.Adam(transformer.parameters(), opt.lr)

    # Load and process style image
    style = utils.get_style_data(opt.style_path)
    if vis:
        vis.img("style", (style.data[0] * 0.225 + 0.45).clamp(min=0, max=1))
    style = style.to(device)

    # Compute style features and gram matrices without gradient computation
    # This is done once at the start since the style image doesn't change
    with t.no_grad():  # Saves memory by not tracking gradients
        # Extract VGG features from style image
        # 
        ## When you call vgg(style), it returns something like this:
        # features_style = vgg(style)

        # features_style is a named tuple containing features from different VGG layers ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        # print("Features from VGG layers:")
        #
        #print("VGG feature shapes:")
        #print(f"relu1_2: {features.relu1_2.shape}")  # [batch_size, 64, 256, 256]
        #print(f"relu2_2: {features.relu2_2.shape}")  # [batch_size, 128, 128, 128]
        #print(f"relu3_3: {features.relu3_3.shape}")  # [batch_size, 256, 64, 64]
        #print(f"relu4_3: {features.relu4_3.shape}")  # [batch_size, 512, 32, 32]
        # features_style contains activations from different VGG layers
        features_style = vgg(style)

        # Compute gram matrices for each feature layer
        # Gram matrix captures style information by measuring feature correlations
        # List comprehension creates a gram matrix for each VGG layer output
        gram_style = [utils.gram_matrix(y) for y in features_style]
        
        # After Gram matrix calculation:
        #gram_style = [utils.gram_matrix(y) for y in features_style]
        #print("\nGram matrix shapes:")
        #print(f"gram1: {gram_style[0].shape}")  # [batch_size, 64, 64]
        #print(f"gram2: {gram_style[1].shape}")  # [batch_size, 128, 128]
        #print(f"gram3: {gram_style[2].shape}")  # [batch_size, 256, 256]
        #print(f"gram4: {gram_style[3].shape}")  # [batch_size, 512, 512]
        
        
        
    # Set up meters to track loss values during training
    # AverageValueMeter maintains running average and standard deviation
    style_meter = tnt.meter.AverageValueMeter()  # Tracks style loss
    content_meter = tnt.meter.AverageValueMeter()  # Tracks content loss

    # Main training loop
    for epoch in range(opt.epoches):
        content_meter.reset()
        style_meter.reset()

        for ii, (x, _) in tqdm.tqdm(enumerate(dataloader)):
            # Move input to device and generate output
            optimizer.zero_grad()
            x = x.to(device)
            y = transformer(x)

            # Normalize images for VGG processing
            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            # Get features from VGG
            features_y = vgg(y)
            features_x = vgg(x)

            # Compute content loss
            content_loss = opt.content_weight * F.mse_loss(
                features_y.relu2_2, features_x.relu2_2
            )

            # Compute style loss
            style_loss = 0.0
            for ft_y, gm_s in zip(features_y, gram_style):
                gram_y = utils.gram_matrix(ft_y)
                style_loss += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
            style_loss *= opt.style_weight

            # Combine losses and update
            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            # Update loss meters
            content_meter.add(content_loss.item())
            style_meter.add(style_loss.item())

            # Visualization and debugging
            if vis and (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                vis.plot("content_loss", content_meter.value()[0])
                vis.plot("style_loss", style_meter.value()[0])
                vis.img("output", (y.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
                vis.img("input", (x.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))

        # Save progress
        if vis:
            vis.save([opt.env])
        t.save(transformer.state_dict(), f"checkpoints/{epoch}_style.pth")


@t.no_grad()
def stylize(**kwargs):
    """Function to apply style transfer to a single image

    Args:
        **kwargs: Overrides for Config parameters
    """
    opt = Config()
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device = t.device("cuda") if opt.use_gpu else t.device("cpu")

    # Load and preprocess input image
    content_image = tv.datasets.folder.default_loader(opt.content_path)
    content_transform = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Lambda(lambda x: x.mul(255))]
    )
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device).detach()

    # Load and initialize model
    style_model = TransformerNet().eval()
    style_model.load_state_dict(
        t.load(opt.model_path, map_location=lambda _s, _: _s, weights_only=True),
        strict=False,
    )
    style_model.to(device)

    # Generate styled image
    output = style_model(content_image)

    # Save result
    output_data = output.cpu().data[0]
    tv.utils.save_image(((output_data / 255)).clamp(min=0, max=1), opt.result_path)


if __name__ == "__main__":
    import fire

    fire.Fire()  # CLI interface for running train() or stylize()
