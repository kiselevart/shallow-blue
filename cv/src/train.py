import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FENChessSquareDataset, ChessPieceCNN

def main():
    # 1. Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Hyperparameters & paths
    data_dir    = "../dataset/train"   # <-- your folder of board .jpeg files
    batch_size  = 64
    lr          = 1e-3
    num_epochs  = 10
    num_workers = os.cpu_count() or 1

    # 3. Prepare dataset
    dataset = FENChessSquareDataset(data_dir)
    # take only a random subset of boards so we finish in ~5min
    subset_boards = 500
    dataset.filenames = random.sample(dataset.filenames, subset_boards)
    print(f"Using {subset_boards} boards → {len(dataset)} total square samples")

    # 4. DataLoader (now sees the correct dataset length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    print(f"DataLoader will use {num_workers} worker processes")

    # 5. Model, loss, optimizer
    model     = ChessPieceCNN(num_classes=13).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 6. Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        loop = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
        for inputs, labels in loop:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # stats
            bs = inputs.size(0)
            running_loss    += loss.item() * bs
            preds            = outputs.argmax(dim=1)
            running_corrects += (preds == labels).sum().item()
            total_samples   += bs

            loop.set_postfix({
                'loss': f"{running_loss/total_samples:.4f}",
                'acc':  f"{running_corrects/total_samples:.4f}"
            })

        epoch_loss = running_loss / len(dataset)
        epoch_acc  = running_corrects / len(dataset)
        print(f"→ Epoch {epoch}: Loss {epoch_loss:.4f} | Acc {epoch_acc:.4f}")

    # 7. Save model weights
    torch.save(model.state_dict(), "chess_piece_cnn.pth")

if __name__ == "__main__":
    main()
