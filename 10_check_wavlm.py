import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from audio_dataset import SpeakerDataset


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model_id = "microsoft/wavlm-base-plus-sv"

# download the feature extractor and model from Hugging Face
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
model = WavLMForXVector.from_pretrained(model_id).to(device)
model.eval()

# read a single test sample to verify the model and feature extractor
dataset = SpeakerDataset(
    split="test",
    duration=4.0,
    snr_db=None,
    random_crop=True,
    return_waveform=True
)

waveform, label = dataset[0]

# waveform is a tensor of shape [1, num_samples], we need to convert it to a list of numpy arrays for the feature extractor
inputs = feature_extractor(
    waveform[0].numpy(),
    sampling_rate=16000,
    return_tensors="pt"
)

inputs = {
    name: tensor.to(device)
    for name, tensor in inputs.items()
}

with torch.no_grad():
    embedding = model(**inputs).embeddings

embedding = torch.nn.functional.normalize(embedding, dim=-1)

print("Device:", device)
print("Speaker label:", label)
print("Waveform shape:", waveform.shape)
print("WavLM embedding shape:", embedding.shape)