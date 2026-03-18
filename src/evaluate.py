"""
evaluate.py
Inference and evaluation for MELD visual stream models.

Aggregates frame-level predictions to video-level via logit averaging,
then computes accuracy, precision, recall, F1, and per-class breakdown.

Usage:
    python src/evaluate.py --model cnn --weights models/best_cnn.pth
    python src/evaluate.py --model vgg --weights models/best_vgg.pth
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt

from models import MELD_CNN, MELD_VGG16, IDX2EMOTION, IDX2SENTIMENT
from dataset import FrameLevelMELDDataset, build_label_dict, get_transform

MELD_BASE = "data/MELD-RAW/MELD.Raw"
FRAMES_BASE = "frames"
BATCH_SIZE = 32


# ---------------------------
# INFERENCE
# ---------------------------

def run_inference(model, loader, device):
    """
    Run model inference on a DataLoader.

    Returns:
        video_sent_preds : list of predicted sentiment labels (video-level)
        video_emo_preds  : list of predicted emotion labels (video-level)
        video_sent_labels: list of true sentiment labels
        video_emo_labels : list of true emotion labels
        sent_probs       : list of sentiment probability arrays
        emo_probs        : list of emotion probability arrays
    """
    model.eval()

    video_sent_logits = defaultdict(list)
    video_emo_logits = defaultdict(list)
    video_sent_labels = {}
    video_emo_labels = {}

    with torch.no_grad():
        for images, (sent_labels, emo_labels, video_ids) in loader:
            images = images.to(device)
            sent_out, emo_out = model(images)

            for i, vid in enumerate(video_ids):
                video_sent_logits[vid].append(sent_out[i].cpu())
                video_emo_logits[vid].append(emo_out[i].cpu())
                video_sent_labels[vid] = sent_labels[i].item()
                video_emo_labels[vid] = emo_labels[i].item()

    # Aggregate by averaging logits across frames
    vids = list(video_sent_logits.keys())
    sent_probs, emo_probs = [], []
    sent_preds, emo_preds = [], []
    true_sent, true_emo = [], []

    for vid in vids:
        s_logits = torch.stack(video_sent_logits[vid]).mean(0)
        e_logits = torch.stack(video_emo_logits[vid]).mean(0)

        s_prob = F.softmax(s_logits, dim=0).numpy()
        e_prob = F.softmax(e_logits, dim=0).numpy()

        sent_probs.append(s_prob)
        emo_probs.append(e_prob)
        sent_preds.append(s_prob.argmax())
        emo_preds.append(e_prob.argmax())
        true_sent.append(video_sent_labels[vid])
        true_emo.append(video_emo_labels[vid])

    return (sent_preds, emo_preds, true_sent, true_emo,
            np.vstack(sent_probs), np.vstack(emo_probs))


# ---------------------------
# METRICS
# ---------------------------

def print_metrics(true_labels, pred_labels, task, idx2label):
    """Print accuracy, precision, recall, F1, and per-class report."""
    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    rec = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

    print(f"\n=== {task} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print("\nPer-class report:")
    target_names = [idx2label[i] for i in sorted(idx2label.keys())]
    print(classification_report(true_labels, pred_labels,
                                 target_names=target_names, zero_division=0))
    return acc, prec, rec, f1


def plot_confusion_matrix(true_labels, pred_labels, idx2label, title, save_path=None):
    """Plot and optionally save a confusion matrix."""
    cm = confusion_matrix(true_labels, pred_labels)
    labels = [idx2label[i] for i in sorted(idx2label.keys())]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
           xticklabels=labels, yticklabels=labels,
           title=title, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix → {save_path}")
    plt.show()


# ---------------------------
# MAIN
# ---------------------------

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.model == "cnn":
        model = MELD_CNN().to(device)
    else:
        model = MELD_VGG16(feature_dim=100).to(device)

    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f"Loaded weights from {args.weights}")

    # Test dataset
    test_labels = build_label_dict(f"{MELD_BASE}/test/test_sent_emo.csv")
    test_ds = FrameLevelMELDDataset(f"{FRAMES_BASE}/test", test_labels,
                                     transform=get_transform(training=False))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Inference
    sent_preds, emo_preds, true_sent, true_emo, sent_probs, emo_probs = \
        run_inference(model, test_loader, device)

    # Metrics
    print_metrics(true_sent, sent_preds, "Sentiment", IDX2SENTIMENT)
    print_metrics(true_emo, emo_preds, "Emotion", IDX2EMOTION)

    # Confusion matrices
    os.makedirs("results", exist_ok=True)
    plot_confusion_matrix(true_sent, sent_preds, IDX2SENTIMENT,
                          f"{args.model.upper()} — Sentiment Confusion Matrix",
                          save_path=f"results/{args.model}_sentiment_cm.png")
    plot_confusion_matrix(true_emo, emo_preds, IDX2EMOTION,
                          f"{args.model.upper()} — Emotion Confusion Matrix",
                          save_path=f"results/{args.model}_emotion_cm.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MELD visual stream model")
    parser.add_argument("--model", choices=["cnn", "vgg"], required=True)
    parser.add_argument("--weights", required=True,
                        help="Path to .pth weights file")
    args = parser.parse_args()
    evaluate(args)
