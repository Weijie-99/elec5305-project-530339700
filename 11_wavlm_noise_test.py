import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from audio_dataset import SpeakerDataset


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model_id = "microsoft/wavlm-base-plus-sv"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
model = WavLMForXVector.from_pretrained(model_id).to(device)
model.eval()


def extract_embeddings(split, snr_db=None):
    dataset = SpeakerDataset(
        split=split,
        duration=4.0,
        snr_db=snr_db,
        random_crop=False,     
        return_waveform=True
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    embeddings = []
    labels_list = []

    with torch.no_grad():
        for waveforms, labels in loader:
            # [batch, 1, samples] → single-channel audio array list
            audio = [
                waveform[0].numpy()
                for waveform in waveforms
            ]

            inputs = feature_extractor(
                audio,
                sampling_rate=16000,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt"
            )

            inputs = {
                name: tensor.to(device)
                for name, tensor in inputs.items()
            }

            embedding = model(**inputs).embeddings

            embeddings.append(embedding.cpu())
            labels_list.append(labels)

    embeddings = F.normalize(
        torch.cat(embeddings), dim=1
    )
    labels = torch.cat(labels_list)

    return embeddings, labels


def approximate_eer(genuine, impostor):
    # calculate the threshold where FAR and FRR are approximately equal
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


# 1. use the enrollment utterances to create reference embeddings for each speaker
print("Extracting enrollment embeddings...", flush=True)

enroll_embeddings, enroll_labels = extract_embeddings("enrollment")

references = torch.stack([
    enroll_embeddings[enroll_labels == label].mean(dim=0)
    for label in range(10)
])

references = F.normalize(references, dim=1)

# 2. stable noise conditions for testing, same as in the CNN test
for snr in [None, 20, 10, 0]:
    condition = "Clean" if snr is None else f"{snr} dB"
    print(f"\nTesting: {condition}", flush=True)

    test_embeddings, test_labels = extract_embeddings(
        "test", snr_db=snr
    )

    scores = test_embeddings @ references.T

    same_speaker = (
        test_labels[:, None] == torch.arange(10)[None, :]
    )

    genuine = scores[same_speaker]
    impostor = scores[~same_speaker]

    eer = approximate_eer(genuine, impostor)

    print(f"Test utterances: {len(test_labels)}")
    print(f"Genuine mean similarity: {genuine.mean().item():.4f}")
    print(f"Impostor mean similarity: {impostor.mean().item():.4f}")
    print(f"Approximate EER: {eer:.2%}")