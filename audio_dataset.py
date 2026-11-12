from pathlib import Path
import json

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader


class SpeakerDataset(Dataset):
    def __init__(
    self, split, duration=4.0, snr_db=None,
    random_crop=False, return_waveform=False
):
        self.return_waveform = return_waveform  
        self.snr_db = snr_db
        self.random_crop = random_crop
        project_dir = Path(__file__).parent
        self.audio_root = project_dir / "data" / "LibriSpeech"

        # read the split information from the JSON file
        with (project_dir / "split_dev_clean.json").open(
            "r", encoding="utf-8"
        ) as file:
            records = json.load(file)

        self.records = [
            item for item in records if item["split"] == split
        ]

        if not self.records:
            raise ValueError(f"没有找到分组：{split}")

        self.training = split == "train"
        self.sample_rate = 16000
        self.target_length = int(duration * self.sample_rate)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=64,
            power=2.0
        )

        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=80
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]

        relative_path = item["path"].replace("\\", "/")
        audio_path = self.audio_root / relative_path

        audio, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=True
        )

        if sample_rate != self.sample_rate:
            raise ValueError(f"采样率不是16000：{audio_path}")

        # [voice, time] → [time, voice] → [voice, time]
        waveform = torch.from_numpy(audio).T
        waveform = waveform.mean(dim=0, keepdim=True)

        length = waveform.shape[-1]

        if length > self.target_length:
            max_start = length - self.target_length

            if self.training:
                # training: random crop, different position each time
                start = torch.randint(0, max_start + 1, (1,)).item()

            elif self.random_crop:
                # test: random crop, but fixed position for each sample
                crop_generator = torch.Generator().manual_seed(2026 + index)

                start = torch.randint(
                    0, max_start + 1, (1,),
                    generator=crop_generator
                ).item()

            else:
                # validation and reference embeddings: keep the original middle crop
                start = max_start // 2

            waveform = waveform[:, start:start + self.target_length]
        elif length < self.target_length:
            # pad waveform to the target length
            waveform = F.pad(
                waveform, (0, self.target_length - length)
            )
        # add noise if snr_db is specified
        if self.snr_db is not None:
            generator = torch.Generator().manual_seed(43 + index)
            noise = torch.randn(waveform.shape, generator=generator)

            signal_power = waveform.square().mean()
            noise_power = noise.square().mean()

            scale = torch.sqrt(
                signal_power
                / (noise_power.clamp_min(1e-12) * 10 ** (self.snr_db / 10))
            )

            waveform = waveform + scale * noise
        # wavlm model expects raw waveform input, so we return the waveform directly if return_waveform is True
        if self.return_waveform:
            return waveform, item["label"]    
        log_mel = self.db_transform(self.mel_transform(waveform))

        # standardize log-mel to have zero mean and unit variance
        log_mel = (
            (log_mel - log_mel.mean())
            / log_mel.std().clamp_min(1e-6)
        )

        return log_mel, item["label"]


# check the dataset and dataloader
if __name__ == "__main__":
    train_dataset = SpeakerDataset(split="train")
    val_dataset = SpeakerDataset(split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    features, labels = next(iter(train_loader))

    print("Train utterances:", len(train_dataset))
    print("Validation utterances:", len(val_dataset))
    print("Batch feature shape:", features.shape)
    print("Batch label shape:", labels.shape)
    print("Labels:", labels)