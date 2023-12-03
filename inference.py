from diffusion_model import SimpleUnet
from utils import plot_denoising
import torch

MAX_TIMESTEP = 1000
IMG_SIZE     = 256

device = "cuda"

model = SimpleUnet().to(torch.device(device))
model.load_state_dict(torch.load("./results/models/model300.pth"))

plot_denoising(model, IMG_SIZE, MAX_TIMESTEP, f"./out.png", device)
