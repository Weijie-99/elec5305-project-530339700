# ELEC5305 Project Proposal

**Teacher:** Craig Jin  
**TA:** Shayikh Hossain and Yu Zhang  

---

## 1 Project Title

**Deep Learning for Speaker Recognition and Speaker Embedding Analysis with Applications to Voice Cloning**

---

## 2 Student Information

- **Full Name:** Weijie Zhang
- **Student ID (SID):** 530339700
- **GitHub Username:** Weijie-99
- **GitHub Project Link:** [elec5305-project-530339700](https://github.com/Weijie-99/elec5305-project-530339700)

---

## 3 Project Overview

This project is motivated by my interest in GPT-SoVITS and voice cloning. The main goal is to study how a deep learning model can learn features that represent different speakers, and to understand how these features may relate to voice cloning.

First, I will train a speaker classification model using speech from several people. The audio will be converted into Log-Mel spectrograms and used as input to a convolutional neural network (CNN), which will learn to identify which speaker produced each speech segment.

After the classifier is trained, the final classification layer will be removed and the feature vector before that layer will be used as a speaker embedding. Embeddings from the same speaker and from different speakers will be compared using cosine similarity.

I will then add different types or levels of noise to the audio and study whether noisy material changes the learned speaker representation. This gives a simple way to connect speaker recognition with a practical problem I have encountered in voice cloning.

---

## 4 Background and Motivation

I have used GPT-SoVITS for a long time when making secondary-creation videos for social media, from early versions to more recent versions. I have trained many voice models and found that the quality of training material can strongly affect the final result.

This made me interested in understanding some of the mechanisms behind the software instead of only using it as a tool.

One practical problem is background noise. It is often difficult to obtain completely clean speech from videos. Sometimes speech with a small amount of background sound can still produce a good model, while in other cases it is better to remove low-quality material.

Manually cleaning audio in Adobe Audition can take a lot of time, and strong or irregular background sounds are also difficult to remove completely. I therefore want to study whether noise changes the speaker features learned by a neural network.

Another issue I have noticed is emotion. For example, a model trained with many different speaking emotions does not always produce a strong angry voice, while a smaller dataset containing mainly angry speech can sometimes work better.

Emotion is interesting, but it may make the project too large. Therefore, the main project will focus on speaker identity and noise. If there is enough time, emotion will be investigated as an optional extension.

Speaker embeddings represent speaker-related information as a numerical vector. Deep neural networks have been widely used to learn these representations for speaker recognition and verification.

Voice-cloning systems also need information about the target speaker, although a complete voice-cloning system requires other components for language, acoustic modelling and waveform generation.

This project focuses only on the speaker representation part so that the scope remains suitable for one semester.

---

## 5 Proposed Methodology

The project will use the following tools, signal-processing methods and data:

- **Tools and platforms:** Python, PyTorch and torchaudio. GPT-SoVITS will be used mainly as background motivation and, if time allows, for a small comparison or demonstration.

- **Signal processing:** waveform preprocessing, short-time Fourier transform (STFT), Mel spectrogram and Log-Mel spectrogram.

- **Deep learning:** a small CNN for speaker classification, followed by extraction of a hidden-layer feature vector as the speaker embedding.

- **Data source:** [LibriSpeech](https://www.openslr.org/12). Approximately 10 speakers with sufficient recordings will be selected and divided into training, validation and testing sets.

- **Evaluation:** classification accuracy and confusion matrix for speaker identification, and cosine similarity for comparing speaker embeddings.

- **Robustness experiment:** compare clean speech with speech containing added white noise or background music at selected signal-to-noise ratios. If time allows, different audio lengths or emotional speech may also be tested.

---

## 6 Expected Outcomes

The expected outputs of the project are:

- A working CNN-based speaker classification model for approximately 10 speakers.
- A target clean-speech classification accuracy of around 85% or higher. This is a project goal rather than a guaranteed result.
- Speaker embeddings extracted from the trained network and analysed using cosine similarity.
- A comparison of clean and noisy audio to show how noise affects classification accuracy and embedding similarity.
- GitHub documentation, source code, experiment results and a final demonstration.
- If time is available, a small optional experiment that relates the learned speaker representation to an open-source voice-generation or voice-cloning system.

---

## 7 Timeline (Weeks 1–13)

| Week | Task |
|---|---|
| 1–2 | Confirm project topic, scope and literature |
| 3–4 | Learn STFT, Log-Mel spectrograms, CNN speaker classification and basic speaker embeddings; prepare LibriSpeech data |
| 5–6 | Implement preprocessing and the first CNN speaker-classification model; complete Report 1 |
| 7–8 | Train, test and adjust the classifier; evaluate clean-speech performance |
| 9 | Extract speaker embeddings and implement cosine-similarity analysis |
| 10–11 | Perform noise experiments and analyse the results; complete Report 2 |
| 12–13 | Prepare final figures, final report, GitHub documentation and demonstration; attempt optional voice-generation experiment if time remains |

---

## 8 References

1. Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015).  
   *LibriSpeech: An ASR corpus based on public domain audio books.*  
   Proceedings of ICASSP 2015, 5206–5210.

2. Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., & Khudanpur, S. (2018).  
   *X-Vectors: Robust DNN embeddings for speaker recognition.*  
   Proceedings of ICASSP 2018, 5329–5333.

3. Wan, L., Wang, Q., Papir, A., & Moreno, I. L. (2018).  
   *Generalized end-to-end loss for speaker verification.*  
   Proceedings of ICASSP 2018, 4879–4883.

4. GPT-SoVITS Project.  
   [GPT-SoVITS: Few-shot voice conversion and text-to-speech WebUI](https://github.com/RVC-Boss/GPT-SoVITS)  
   Technical reference; not counted as one of the three peer-reviewed references.

---

## 9 Appendix (Optional)

### Planned Project Workflow

```text
Audio waveform
      ↓
STFT / Log-Mel Spectrogram
      ↓
CNN
      ↓
Speaker Embedding
      ↓
Speaker Classification
      +
Cosine-Similarity Analysis
