from pathlib import Path
from torchaudio.datasets import LIBRISPEECH


data_dir = Path(__file__).parent / "data"

data_dir.mkdir(parents=True, exist_ok=True)

dataset = LIBRISPEECH(
    root=data_dir,
    url="dev-clean",
    download=True
)

print("音频总数量：", len(dataset))

audio_path, sample_rate, transcript, speaker_id, chapter_id, utterance_id = \
    dataset.get_metadata(0)

print("音频路径：", audio_path)
print("采样率：", sample_rate)
print("文本内容：", transcript)
print("说话人ID：", speaker_id)
print("章节ID：", chapter_id)
print("音频ID：", utterance_id)