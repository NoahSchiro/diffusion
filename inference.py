from diffusion_model import SimpleUnet
from utils import plot_denoising
import torch

MAX_TIMESTEP = 300
BATCH_SIZE   = 128
IMG_SIZE     = 64
NUM_EPOCHS   = 100
STATS        = (0.5229942, 0.48899996, 0.41180329), (0.25899375, 0.24669976, 0.25502672)

device = "cuda"

model = SimpleUnet().to(torch.device(device))
model.load_state_dict(torch.load("./results/models/model4.pth"))

plot_denoising(model, IMG_SIZE, MAX_TIMESTEP, f"./out.png", device)
