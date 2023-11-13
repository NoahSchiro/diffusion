# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_model import SimpleUnet, get_loss
from utils import (
    load_transformed_dataset,
    plot_denoising,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TIMESTEP = 300
BATCH_SIZE   = 128
IMG_SIZE     = 64
NUM_EPOCHS   = 100
STATS        = (0.5229942, 0.48899996, 0.41180329), (0.25899375, 0.24669976, 0.25502672)

if __name__=="__main__":
    data = load_transformed_dataset(IMG_SIZE, STATS)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = SimpleUnet()
    print("Loaded model")

    # Training
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHS):
        for step, batch in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()

            batch = batch[0]

            t = torch.randint(0, MAX_TIMESTEP, (BATCH_SIZE,), device=DEVICE).long()

            loss = get_loss(model, batch, t, DEVICE)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                plot_denoising(model, IMG_SIZE, MAX_TIMESTEP, f"./results/figures/figure{epoch}.png", DEVICE)

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"./results/models/model{epoch}.pth")
