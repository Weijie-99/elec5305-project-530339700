import torch
from torch import nn


class SpeakerCNN(nn.Module):
    def __init__(self, num_speakers=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((16, 1)),
            nn.Flatten()
        )

        self.embedding_layer = nn.Linear(128 * 16, 64)
        self.classifier = nn.Linear(64, num_speakers)

    def forward(self, x):
        features = self.features(x)
        embedding = self.embedding_layer(features)
        embedding = torch.relu(embedding)
        logits = self.classifier(embedding)

        return logits, embedding

