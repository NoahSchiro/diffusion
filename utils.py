import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from diffusion_model import forward_diffusion_sample, sample_timestep

from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.transforms import functional as F

# I need to create a custom transform
class ResizeAspectRatio:
    def __init__(self, targetWidth, targetHeight):
        self.size = (targetWidth, targetHeight) 

    def __call__(self, img):

        w, h = img.size

        aspect_ratio = float(h) / float(w)

        # If aspect ratio is greater than 1, this means height
        # is larger than width (i.e. a portrait). So the width
        # is the lower bound
        if aspect_ratio > 1:

            # Resize to this
            img = F.resize(img, self.size[0])
        else:
            # Resize to this
            img = F.resize(img, self.size[1])

        # Crop down from the top left corner
        return F.crop(img, 0, 0, self.size[1], self.size[0])
 
def load_transformed_dataset(size, stats):
    targetw, targeth = size, size 

    transform = T.Compose([
        ResizeAspectRatio(targetw, targeth),
        T.ToTensor(),
        T.Normalize(*stats)
    ])

    return ImageFolder("./data", transform)

def show_tensor_image(image):
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))


@torch.no_grad()
def plot_denoising(model, img_size, max_timestep, save_file, device="cpu"):
    # Sample noise
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.title("Current model de-noising for decreasing time t", y=1.08)
    num_images = 9
    subplot_dim = math.ceil(math.sqrt(num_images))
    plot_steps = np.floor(np.linspace(0, max_timestep - 1, num_images))

    subplot_idx = 0
    for time_val in range(0, max_timestep)[::-1]:
        t = torch.full((1,), time_val, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)

        if time_val in plot_steps:
            ax = plt.subplot(subplot_dim, subplot_dim, subplot_idx + 1)
            ax.set_title(f"t={time_val}")
            show_tensor_image(img.detach().cpu())
            subplot_idx += 1
    plt.savefig(save_file)
