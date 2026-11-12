from pathlib import Path
from cnn_model import SpeakerCNN
import matplotlib.pyplot as plt
import torchaudio
import torch
import soundfile as sf

# 1. use torchaudio.datasets.LIBRISPEECH to load the dataset
data_dir = Path(__file__).parent / "data"

dataset = torchaudio.datasets.LIBRISPEECH(
    root=str(data_dir),
    url="dev-clean",
    download=False
)

# 2. read the first audio file and convert it to a waveform tensor
# rather than using the built-in waveform loading, we will manually read the audio file to demonstrate the process
relative_path, sample_rate, text, speaker_id, chapter_id, utterance_id = (
    dataset.get_metadata(16)
)

# construct the full path to the audio file
audio_path = data_dir / "LibriSpeech" / relative_path

# read the audio file using soundfile (sf) to get a NumPy array and the sample rate
audio, sample_rate = sf.read(
    str(audio_path),
    dtype="float32",
    always_2d=True
)

# transpose the audio array to have shape (channels, samples) and convert it to a PyTorch tensor
waveform = torch.from_numpy(audio).T

print("Speaker ID:", speaker_id)
print("Sample rate:", sample_rate)
print("Waveform shape:", waveform.shape)
print("Duration:", waveform.shape[-1] / sample_rate, "seconds")

# 3. create a MelSpectrogram transform and apply it to the waveform to get the Mel spectrogram
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=64,
    power=2.0
)

# 4. apply the MelSpectrogram transform to the waveform and then convert it to a log-Mel spectrogram
mel = mel_transform(waveform)

db_transform = torchaudio.transforms.AmplitudeToDB(
    stype="power",
    top_db=80
)

log_mel = db_transform(mel)

print("Log-Mel shape:", log_mel.shape)

# [1, 64, T] → [1, 1, 64, T]
cnn_input = log_mel.unsqueeze(0)

model = SpeakerCNN(num_speakers=10)
model.eval()

with torch.no_grad():
    logits, embedding = model(cnn_input)

print("CNN input shape:", cnn_input.shape)
print("Classification output shape:", logits.shape)
print("Embedding shape:", embedding.shape)

# 5. display the Log-Mel spectrogram for the first channel
plt.figure(figsize=(10, 4))

plt.imshow(
    log_mel[0].numpy(),
    origin="lower",
    aspect="auto",
    cmap="magma"
)

plt.xlabel("Time frame")
plt.ylabel("Mel bin")
plt.title(f"Log-Mel Spectrogram — Speaker {speaker_id}")
plt.colorbar(label="Power (dB)")
plt.tight_layout()
plt.show()

