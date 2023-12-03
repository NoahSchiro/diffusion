import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
#from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
import matplotlib.pyplot as plt

from diffusion_model import SimpleUnet, get_loss
from utils import (
    load_transformed_dataset,
    plot_denoising,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TIMESTEP = 1000 
BATCH_SIZE   = 24 
IMG_SIZE     = 256 
NUM_EPOCHS   = 300
LR           = 1e-3 
STATS        = (0.5229942, 0.48899996, 0.41180329), (0.25899375, 0.24669976, 0.25502672)

#scaler = GradScaler()

def train(model, optimizer, epoch):
    for batch, imgs in enumerate(dataloader):
        optimizer.zero_grad()

        imgs = imgs[0]

        t = torch.randint(0, MAX_TIMESTEP, (BATCH_SIZE,), device=DEVICE).long()

        #with autocast():
        loss = get_loss(model, imgs, t, DEVICE)
        
        loss.backward()
        optimizer.step()
        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()

        if batch % 10 == 0:
            print(f"Epoch {epoch} | batch {batch:02d}/{len(dataloader)} | Loss: {loss.item()} ")

        if epoch % 5 == 0 and batch == 0:
            plot_denoising(model, IMG_SIZE, MAX_TIMESTEP, f"./results/figures/figure{epoch}.png", DEVICE)
            torch.save(model.state_dict(), f"./results/models/model{epoch}.pth")
    

if __name__=="__main__":
    data = load_transformed_dataset(IMG_SIZE, STATS)
    dataloader = DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=12,
        pin_memory=True
    )

    model = SimpleUnet()
    print(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters")

    # Training
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS+1):
        train(model, optimizer, epoch)
