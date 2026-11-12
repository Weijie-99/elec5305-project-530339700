from pathlib import Path
from datetime import datetime
import json
import random
import math
import torch
from torch import nn
from torch.utils.data import DataLoader

from audio_dataset import SpeakerDataset
from cnn_model import SpeakerCNN


def run_epoch(model, loader, criterion, device, optimizer=None,scheduler=None):
    # training mode if optimizer is provided, otherwise evaluation mode
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        # disable gradient calculation for validation to save memory and computation
        with torch.set_grad_enabled(training):
            logits, embedding = model(features)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        # record the number of samples and correct predictions for accuracy calculation
        batch_size = labels.size(0)
        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )

def build_scheduler(optimizer, warmup_steps, total_steps):
    # Linear warm-up, then cosine decay to zero, updated after every batch.
    # The warm-up keeps the large early gradients of the new classifier from
    # destroying the pre-trained features.
    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)



def main():
    # use fixed random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # create a unique output directory for this run
    project_dir = Path(__file__).parent
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = project_dir / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    train_dataset = SpeakerDataset(split="train", duration=4.0)
    val_dataset = SpeakerDataset(split="val", duration=4.0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    print("Train utterances:", len(train_dataset))
    print("Validation utterances:", len(val_dataset))

    model = SpeakerCNN(num_speakers=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 60
    # warm-up
    warmup_steps = 3 * len(train_loader)
    total_steps = epochs * len(train_loader)

    scheduler = build_scheduler(
        optimizer, warmup_steps, total_steps
    )
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    history = []

    # save the split information to the output directory for reference
    split_path = project_dir / "split_dev_clean.json"
    split_text = split_path.read_text(encoding="utf-8")
    (output_dir / "split.json").write_text(
        split_text, encoding="utf-8"
    )

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scheduler
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.1%} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.1%}"
        )

        # optimize for best validation accuracy, and if tied, choose the one with lower validation loss
        improved = (
            val_acc > best_val_acc
            or (val_acc == best_val_acc and val_loss < best_val_loss)
        )

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "num_speakers": 10,
                "duration": 4.0,
                "seed": 42,
                "learning_rate": 0.001,
                "batch_size": 16
            }, output_dir / "best_cnn.pt")

            print("  Saved best model")

        # save the training history to a JSON file after each epoch
        with (output_dir / "history.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(history, file, indent=2)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_acc:.1%}")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()