from pathlib import Path
from collections import Counter
from torchaudio.datasets import LIBRISPEECH


data_dir = Path(__file__).parent / "data"

dataset = LIBRISPEECH(
    root=data_dir,
    url="dev-clean",
    download=False
)

speaker_counts = Counter()
speaker_chapters = {}

for index in range(len(dataset)):
    metadata = dataset.get_metadata(index)

    speaker_id = metadata[3]
    chapter_id = metadata[4]

    # store the count of audio files for each speaker
    speaker_counts[speaker_id] += 1

    # create a set to store chapter IDs for each speaker if it doesn't exist
    if speaker_id not in speaker_chapters:
        speaker_chapters[speaker_id] = set()

    # record the chapter ID for the speaker
    speaker_chapters[speaker_id].add(chapter_id)


print("说话人总数量：", len(speaker_counts))
print("\n音频数量最多的10名说话人：")

selected_speakers = []

for speaker_id, audio_count in speaker_counts.most_common(10):
    selected_speakers.append(speaker_id)

    print(
        "说话人ID：", speaker_id,
        "音频数量：", audio_count,
        "章节数量：", len(speaker_chapters[speaker_id]),
        "章节ID：", sorted(speaker_chapters[speaker_id])
    )

print("\n最终选择的说话人：")
print(selected_speakers)