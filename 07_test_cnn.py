from pathlib import Path

import torch
from torch.utils.data import DataLoader

from audio_dataset import SpeakerDataset
from cnn_model import SpeakerCNN


project_dir = Path(__file__).parent

# change this path to the actual checkpoint path on your system
checkpoint_path = Path(
    r"C:\Users\wswj1\Desktop\5305\elec5305-project-530339700\runs\20261112_033508_093145\best_cnn.pt"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 1. create the test dataset and DataLoader
test_dataset = SpeakerDataset(split="test", duration=4.0)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# 2. load the model checkpoint and initialize the model
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

model = SpeakerCNN(num_speakers=10).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 3. predict on the test set and calculate accuracy
correct = 0
total = 0

with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)

        logits, embedding = model(features)
        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print("Checkpoint epoch:", checkpoint["epoch"])
print("Test utterances:", total)
print(f"Correct: {correct}/{total}")
print(f"Test accuracy: {correct / total:.2%}")