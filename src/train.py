"""Minimal training script for 2.5D cryo-ET membrane segmentation"""

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from data import CryoETDataset 
from model import UNet

import matplotlib.pyplot as plt


def get_loaders():
    train_ds = CryoETDataset(split="train", length=config.TRAIN_LENGTH)
    val_ds = CryoETDataset(split="val",   length=config.VAL_LENGTH)
    test_ds = CryoETDataset(split="test", length=config.VAL_LENGTH)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        drop_last=False,
    )



    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> float:
    model.eval()
    total = 0.0
    n = 0
    for X, Y in loader:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        logits = model(X)
        loss = criterion(logits, Y)
        total += float(loss.item())
        n += 1
    return total / max(n, 1)



class BCE_Dice_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = config.BCE_WEIGHT

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        bce = self.bce(logits, targets)

        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs * targets_flat).sum(dim=1)
        union = probs.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + 1.0) / (union + 1.0)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss



def train(model: nn.Module, criterion: nn.Module, train_loader: DataLoader, val_loader: DataLoader):
    print("trainin:")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    
    train_losses = []
    val_losses = []
    running_train_loss = 0.0
    
    best_loss = float("inf")
    best_state = None
    
    for epoch in range(1, config.EPOCHS + 1):
        model.train()

        for step, (X, Y) in enumerate(train_loader, start=1):
            X = X.to(config.DEVICE, non_blocking=True)
            Y = Y.to(config.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, Y)
            loss.backward()
            optimizer.step()
            
            running_train_loss += float(loss.item())

            if step % 50 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")
        
        train_losses.append(running_train_loss/len(train_loader))
        running_train_loss = 0.0
        val_loss = evaluate(model, val_loader, criterion, config.DEVICE)
        val_losses.append(val_loss)
        
        if (val_loss < best_loss):
            best_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        print(f"finished epoch={epoch}")
    
    return train_losses, val_losses, best_state, best_loss



def test(model: nn.Module, criterion: nn.Module, test_loader: DataLoader, best_state: dict) -> float:
    model.load_state_dict(best_state, strict=True)
    model = model.to(config.DEVICE)

    test_loss = evaluate(model, test_loader, criterion, config.DEVICE)
    print(f"test_loss={test_loss:.4f}")

    return test_loss


    
def plot_losses(train_losses, val_losses, epochs):
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.show()



def main():
    train_loader, val_loader, test_loader = get_loaders()
    model = UNet().to(config.DEVICE)

    criterion = BCE_Dice_Loss()
    train_losses, val_losses, best_state, best_loss = train(model, criterion, train_loader, val_loader)
    
    x_axis = [i for i in range(1, config.EPOCHS + 1)]
    
    plot_losses(train_losses, val_losses, x_axis)
    
    os.makedirs(config.SAVE, exist_ok=True)
    torch.save(best_state, f"{config.SAVE}/best_weights.pt")

    test_loss = test(model, criterion, test_loader, best_state)
    
    with open(f'{config.SAVE}/report.txt', 'w') as f:
        f.write(f"Best validation loss: {best_loss:.4f}\n")
        f.write(f"Testing loss: {test_loss:.4f}\n")

if __name__ == "__main__":
    main()

