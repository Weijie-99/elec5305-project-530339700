from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from audio_dataset import SpeakerDataset
from cnn_model import SpeakerCNN


# use the same checkpoint path as in 07_test_cnn.py
project_dir = Path(__file__).parent
checkpoint_path = (
    project_dir / "runs" / "20261112_033508_093145" / "best_cnn.pt"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

model = SpeakerCNN(num_speakers=10).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def extract_embeddings(split, duration):
    dataset = SpeakerDataset(
        split=split,
        duration=duration,
        snr_db=None,
        random_crop=(split == "test")
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    embeddings = []
    labels_list = []
    correct = 0

    with torch.no_grad():
        for features, labels in loader:
            logits, embedding = model(features.to(device))

            predictions = logits.argmax(dim=1).cpu()
            correct += (predictions == labels).sum().item()

            embeddings.append(embedding.cpu())
            labels_list.append(labels)

    embeddings = F.normalize(
        torch.cat(embeddings), dim=1
    )
    labels = torch.cat(labels_list)

    return embeddings, labels, correct / len(dataset)


def approximate_eer(genuine, impostor):
    thresholds = torch.unique(torch.cat([genuine, impostor]))
    thresholds = torch.cat([
        torch.tensor([-float("inf")]),
        thresholds,
        torch.tensor([float("inf")])
    ])

    # rate of impostor being accepted as genuine
    far = (
        impostor[:, None] >= thresholds[None, :]
    ).float().mean(dim=0)

    # rate of genuine being rejected as impostor
    frr = (
        genuine[:, None] < thresholds[None, :]
    ).float().mean(dim=0)

    index = (far - frr).abs().argmin()
    return ((far[index] + frr[index]) / 2).item()


# 1. create reference embeddings for each speaker by averaging their enrollment utterances
enroll_embeddings, enroll_labels, _ = extract_embeddings(
    "enrollment", duration=4.0
)

references = torch.stack([
    enroll_embeddings[enroll_labels == label].mean(dim=0)
    for label in range(10)
])

references = F.normalize(references, dim=1)

# 2. change the duration of test utterances while keeping the reference embeddings fixed
for duration in [4.0, 2.0, 1.0, 0.5]:
    test_embeddings, test_labels, accuracy = extract_embeddings(
        "test", duration=duration
    )

    scores = test_embeddings @ references.T

    same_speaker = (
        test_labels[:, None] == torch.arange(10)[None, :]
    )

    genuine = scores[same_speaker]
    impostor = scores[~same_speaker]

    eer = approximate_eer(genuine, impostor)

    print(f"\nDuration: {duration} seconds")
    print(f"Test utterances: {len(test_labels)}")
    print(f"Classification accuracy: {accuracy:.2%}")
    print(f"Genuine mean similarity: {genuine.mean().item():.4f}")
    print(f"Impostor mean similarity: {impostor.mean().item():.4f}")
    print(f"Approximate EER: {eer:.2%}")