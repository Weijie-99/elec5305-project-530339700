from pathlib import Path
from collections import defaultdict
import json
import random

import torchaudio


project_dir = Path(__file__).parent
data_dir = project_dir / "data"

dataset = torchaudio.datasets.LIBRISPEECH(
    root=str(data_dir),
    url="dev-clean",
    download=False
)

# 1.define a dictionary to group audio files by speaker ID
speaker_files = defaultdict(list)

for index in range(len(dataset)):
    path, sample_rate, text, speaker_id, chapter_id, utterance_id = (
        dataset.get_metadata(index)
    )

    speaker_files[speaker_id].append({
        "path": path,
        "speaker_id": speaker_id,
        "chapter_id": chapter_id
    })

# 2. choose speakers with at least 40 audio files, sort them, and select the first 10
selected_speakers = sorted(
    speaker_id
    for speaker_id, files in speaker_files.items()
    if len(files) >= 40
)[:10]

if len(selected_speakers) < 10:
    raise ValueError("符合条件的说话人不足10位")

# seed the random number generator for reproducibility
rng = random.Random(42)

records = []

# 3. split the audio files for each selected speaker into train, val, enrollment, and test sets
for label, speaker_id in enumerate(selected_speakers):
    files = speaker_files[speaker_id].copy()
    rng.shuffle(files)

    total = len(files)

    train_end = int(total * 0.60)
    val_end = train_end + int(total * 0.15)
    enrollment_end = val_end + int(total * 0.10)

    groups = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "enrollment": files[val_end:enrollment_end],
        "test": files[enrollment_end:]
    }

    print(f"Speaker {speaker_id} → label {label}")

    for split_name, split_files in groups.items():
        print(f"  {split_name}: {len(split_files)}")

        for item in split_files:
            records.append({
                **item,
                "label": label,
                "split": split_name
            })

# 4. check for duplicate audio files across splits
paths = [item["path"] for item in records]
assert len(paths) == len(set(paths)), "发现重复音频！"

# 5. save the split information to a JSON file
output_path = project_dir / "split_dev_clean.json"

with output_path.open("w", encoding="utf-8") as file:
    json.dump(records, file, indent=2, ensure_ascii=False)

print(f"\nTotal utterances: {len(records)}")
print(f"Saved to: {output_path}")