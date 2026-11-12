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
    checkpoint_path, map_location=device, weights_only=True
)

model = SpeakerCNN(num_speakers=10).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def extract_embeddings(split, snr_db=None):
    dataset = SpeakerDataset(
        split=split, duration=4.0, snr_db=snr_db
    )

    loader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=0
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

    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels_list)

    # use L2 normalization to make the embeddings unit vectors, which is common in speaker verification tasks
    embeddings = F.normalize(embeddings, dim=1)

    return embeddings, labels, correct / len(dataset)


def approximate_eer(genuine, impostor):
    # try to find the threshold where FAR and FRR are approximately equal
    thresholds = torch.unique(torch.cat([genuine, impostor]))
    thresholds = torch.cat([
        torch.tensor([-float("inf")]),
        thresholds,
        torch.tensor([float("inf")])
    ])


    far = (
        impostor[:, None] >= thresholds[None, :]
    ).float().mean(dim=0)

    frr = (
        genuine[:, None] < thresholds[None, :]
    ).float().mean(dim=0)

    index = (far - frr).abs().argmin()
    return ((far[index] + frr[index]) / 2).item()


# 1. each speaker has 5 enrollment utterances, we will average their embeddings to create a reference embedding for each speaker
enroll_embeddings, enroll_labels, _ = extract_embeddings("enrollment")

references = torch.stack([
    enroll_embeddings[enroll_labels == label].mean(dim=0)
    for label in range(10)
])

references = F.normalize(references, dim=1)

# 2. stable noise conditions for testing, same as in the CNN test
for snr in [None, 20, 10, 0]:
    test_embeddings, test_labels, accuracy = extract_embeddings(
        "test", snr_db=snr
    )

    # [117, 64] @ [64, 10] → [117, 10]
    scores = test_embeddings @ references.T

    same_speaker = (
        test_labels[:, None] == torch.arange(10)[None, :]
    )

    genuine = scores[same_speaker]
    impostor = scores[~same_speaker]

    eer = approximate_eer(genuine, impostor)
    condition = "Clean" if snr is None else f"{snr} dB"

    print(f"\nCondition: {condition}")
    print(f"Classification accuracy: {accuracy:.2%}")
    print(f"Genuine mean similarity: {genuine.mean().item():.4f}")
    print(f"Impostor mean similarity: {impostor.mean().item():.4f}")
    print(f"Approximate EER: {eer:.2%}")