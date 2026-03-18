"""
models.py
Visual stream model architectures for MELD emotion/sentiment classification.

  - MELD_CNN    : Custom 5-layer CNN with Leaky ReLU (baseline)
  - MELD_VGG16  : Pretrained VGG-16 with dual-head output (improved)

Both models output two heads simultaneously:
  - Sentiment: 3 classes (neutral, positive, negative)
  - Emotion:   7 classes (neutral, anger, disgust, sadness, joy, surprise, fear)
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ============================================================
# BASELINE — Custom 5-Layer CNN
# ============================================================

class MELD_CNN(nn.Module):
    """
    5-layer custom CNN for frame-level emotion and sentiment classification.

    Architecture:
        Conv1 (3→16, 3×3) + LeakyReLU + MaxPool + Dropout
        Conv2 (16→32, 3×3) + LeakyReLU + BN + MaxPool + Dropout
        Conv3 (32→64, 3×3) + LeakyReLU + MaxPool + Dropout
        Conv4 (64→128, 3×3) + LeakyReLU + MaxPool + Dropout
        Conv5 (128→256, 3×3) + LeakyReLU + MaxPool
        FC + LeakyReLU → Sentiment head (3) + Emotion head (7)

    Input:  (B, 3, 224, 224)
    Output: (sentiment_logits, emotion_logits)
    """

    def __init__(self, num_sentiment_classes=3, num_emotion_classes=7):
        super(MELD_CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.leakyrelu1 = nn.LeakyReLU(0.1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.leakyrelu2 = nn.LeakyReLU(0.1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.leakyrelu3 = nn.LeakyReLU(0.1)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.2)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.leakyrelu4 = nn.LeakyReLU(0.1)
        self.maxpool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.2)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.leakyrelu5 = nn.LeakyReLU(0.1)
        self.maxpool5 = nn.MaxPool2d(2, 2)

        # Fully connected shared layer
        # After 5× MaxPool2d(2,2) on 224×224: 224/32 = 7 → 7×7×256 = 12,544
        self.fc = nn.Linear(256 * 7 * 7, 512)
        self.leakyrelu_fc = nn.LeakyReLU(0.1)
        self.dropout_fc = nn.Dropout(0.5)

        # Dual output heads
        self.sentiment_head = nn.Linear(512, num_sentiment_classes)
        self.emotion_head = nn.Linear(512, num_emotion_classes)

    def forward(self, x):
        x = self.dropout1(self.maxpool1(self.leakyrelu1(self.conv1(x))))
        x = self.dropout2(self.maxpool2(self.batchnorm1(self.leakyrelu2(self.conv2(x)))))
        x = self.dropout3(self.maxpool3(self.leakyrelu3(self.conv3(x))))
        x = self.dropout4(self.maxpool4(self.leakyrelu4(self.conv4(x))))
        x = self.maxpool5(self.leakyrelu5(self.conv5(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout_fc(self.leakyrelu_fc(self.fc(x)))

        sentiment = self.sentiment_head(x)
        emotion = self.emotion_head(x)
        return sentiment, emotion


# ============================================================
# IMPROVED — Pretrained VGG-16
# ============================================================

class MELD_VGG16(nn.Module):
    """
    Pretrained VGG-16 (ImageNet) fine-tuned for MELD emotion/sentiment.

    Architecture:
        VGG-16 feature extractor (pre-trained, all layers trainable)
        → AdaptiveAvgPool
        → Flatten
        → FC (512×7×7 → feature_dim) + LeakyReLU
        → Sentiment head (feature_dim → 3)
        → Emotion head   (feature_dim → 7)

    Input:  (B, 3, 224, 224) — ImageNet-normalized
    Output: (sentiment_logits, emotion_logits)
    """

    def __init__(self, feature_dim=100, num_sentiment_classes=3, num_emotion_classes=7):
        super(MELD_VGG16, self).__init__()

        vgg16 = models.vgg16_bn(pretrained=True)

        # Convolutional feature extractor
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool     # AdaptiveAvgPool2d(7, 7)
        self.flatten = nn.Flatten()

        # Compact feature projection
        self.fc1 = nn.Linear(512 * 7 * 7, feature_dim)
        self.leakyrelu = nn.LeakyReLU(0.1)

        # Dual output heads
        self.sentiment_head = nn.Linear(feature_dim, num_sentiment_classes)
        self.emotion_head = nn.Linear(feature_dim, num_emotion_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.leakyrelu(self.fc1(x))

        sentiment = self.sentiment_head(x)
        emotion = self.emotion_head(x)
        return sentiment, emotion


# ============================================================
# LABEL MAPPINGS
# ============================================================

EMOTION2IDX = {
    'neutral': 0, 'anger': 1, 'disgust': 2,
    'sadness': 3, 'joy': 4, 'surprise': 5, 'fear': 6
}
IDX2EMOTION = {v: k for k, v in EMOTION2IDX.items()}

SENTIMENT2IDX = {'neutral': 0, 'positive': 1, 'negative': 2}
IDX2SENTIMENT = {v: k for k, v in SENTIMENT2IDX.items()}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 224, 224).to(device)

    print("=== MELD_CNN ===")
    cnn = MELD_CNN().to(device)
    s, e = cnn(x)
    print(f"Sentiment: {s.shape} | Emotion: {e.shape}")

    print("\n=== MELD_VGG16 ===")
    vgg = MELD_VGG16().to(device)
    s, e = vgg(x)
    print(f"Sentiment: {s.shape} | Emotion: {e.shape}")
