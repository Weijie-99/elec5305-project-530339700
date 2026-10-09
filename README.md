# From Speaker Recognition to Voice Cloning: Robustness of Speaker Embeddings to Reference-Audio Degradation

ELEC5305 project by Weijie Zhang (SID 530339700)

## Project overview

I have used GPT-SoVITS for voice cloning and noticed that the quality of reference recordings can change the result. Speech from videos may contain background noise, and the usable clean segment may be short. This project studies how these conditions affect speaker embeddings: numerical representations of speaker identity.

**Research question:** How robust are speaker embeddings to noise and short speech duration, and how does a small task-trained CNN compare with a pretrained speaker-verification model?

Voice cloning motivates the project, but the main experiment concerns speaker representation and verification. A small GPT-SoVITS experiment is optional if the main analysis is complete.

## Models and signal processing

| Model | Input | Purpose |
| --- | --- | --- |
| Student-trained CNN | Log-Mel spectrogram | Train a speaker classifier; take the feature vector before its final classification layer as an embedding |
| [WavLM-Base-Plus-SV](https://huggingface.co/microsoft/wavlm-base-plus-sv) | 16 kHz audio waveform | Extract a pretrained speaker embedding without training WavLM |

For the CNN, the processing path is waveform → framing and STFT → Mel filtering → log compression → CNN → embedding. Classification accuracy will be a supporting result; the main comparison concerns embedding stability and speaker verification.

## Data and experiment plan

I am using [LibriSpeech](https://www.openslr.org/12). The smaller `dev-clean` subset is for initial data and code checks. For the main experiment, I plan to select about 10–20 speakers with sufficient recordings from suitable LibriSpeech training data. CNN training, validation, enrollment, and test utterances will be separate. Where possible, I will also separate chapters to reduce shared recording conditions.

Several clean enrollment utterances will form a reference for each speaker. Different test utterances will make genuine trials (same speaker) and impostor trials (different speakers). I will check clean-speech performance before adding degradation.

1. **Noise robustness:** Fix the test duration and compare clean speech with conditions near 20, 10, 5, and 0 dB SNR. Real environmental noise is the main planned condition; white noise and music are possible comparisons.
2. **Duration robustness:** Compare test segments around 0.5, 1, 2, and 4 seconds, with full utterances where suitable. The cropping method will be documented.

The initial CNN will be trained on clean speech so the main test measures its response to degradation. Noise augmentation is a possible later extension.

## Evaluation

Both models will use the same test trials. Planned results include:

- Cosine similarity between clean and degraded versions of the **same** utterance, to measure embedding stability.
- Genuine and impostor cosine-score distributions from **different** utterances, to measure speaker discrimination.
- Equal error rate (EER), where practical, and CNN classification accuracy as a supporting result.
- Selected Log-Mel examples to relate score changes to changes in the audio signal.

Comparing a recording with its own noisy copy measures stability, but does not by itself measure speaker-verification performance.

## Current progress

- Configured Python, PyTorch, and torchaudio with CUDA support locally.
- Downloaded and inspected LibriSpeech `dev-clean`: 2,703 utterances from 40 speakers at 16 kHz.
- Wrote local scripts to inspect utterance metadata, speaker counts, and chapter IDs.
- Defined an initial small CNN design with a classification output and a 64-dimensional feature vector. It has **not yet been trained or evaluated**.

These are preliminary setup and code checks. CNN accuracy, embedding robustness, and EER have not yet been measured.

## Next steps

1. Select speakers and prepare a reproducible data split.
2. Generate and inspect Log-Mel spectrograms from real audio.
3. Train and validate the CNN on clean speech.
4. Extract CNN and WavLM embeddings for the same trials.
5. Evaluate clean verification, noise robustness, and duration robustness.
6. Prepare plots, examples, and the final report.

If time permits, I will run a small downstream experiment using the same pretrained GPT-SoVITS system with clean and degraded reference clips and fixed synthesis text. Generated speech can then be compared with clean target-speaker speech using an independent speaker model. This extension will test, rather than assume, whether embedding instability relates to cloning results.

## References and software

- Panayotov et al. (2015), *LibriSpeech: An ASR corpus based on public domain audio books*.
- Snyder et al. (2018), *X-Vectors: Robust DNN embeddings for speaker recognition*.
- Wan et al. (2018), *Generalized end-to-end loss for speaker verification*.
- [Microsoft WavLM speaker-verification model](https://huggingface.co/microsoft/wavlm-base-plus-sv).
- [GPT-SoVITS repository](https://github.com/RVC-Boss/GPT-SoVITS).
