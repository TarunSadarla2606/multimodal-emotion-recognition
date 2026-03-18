"""
train.py
Training loop for MELD visual stream models (CNN baseline or VGG-16 improved).

Usage:
    python src/train.py --model cnn
    python src/train.py --model vgg --epochs 30 --lr 0.0005
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from models import MELD_CNN, MELD_VGG16
from dataset import FrameLevelMELDDataset, build_label_dict, get_transform

# ---------------------------
# DEFAULT CONFIG
# ---------------------------
MELD_BASE = "data/MELD-RAW/MELD.Raw"
FRAMES_BASE = "frames"
MODEL_DIR = "models"

BATCH_SIZE = 32
LR = 0.001
EPOCHS = 50
PATIENCE = 5          # early stopping patience
MIN_DELTA = 0.001


# ---------------------------
# VIDEO-LEVEL AGGREGATION
# ---------------------------

def aggregate_by_majority_vote(preds_dict):
    """Aggregate frame-level predictions to video-level by majority vote."""
    return {vid: Counter(preds).most_common(1)[0][0]
            for vid, preds in preds_dict.items()}


# ---------------------------
# ONE EPOCH
# ---------------------------

def run_epoch(model, loader, optimizer, sent_criterion, emo_criterion,
              device, training=True):
    """Run one training or validation epoch. Returns (sent_loss, emo_loss, sent_acc, emo_acc)."""
    model.train() if training else model.eval()

    sent_losses, emo_losses = [], []
    video_sent_preds = defaultdict(list)
    video_emo_preds = defaultdict(list)
    video_sent_labels = {}
    video_emo_labels = {}

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, (sent_labels, emo_labels, video_ids) in tqdm(loader, leave=False):
            images = images.to(device)
            sent_labels = sent_labels.to(device)
            emo_labels = emo_labels.to(device)

            sent_out, emo_out = model(images)

            sent_loss = sent_criterion(sent_out, sent_labels)
            emo_loss = emo_criterion(emo_out, emo_labels)
            total_loss = sent_loss + emo_loss

            if training:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            sent_losses.append(sent_loss.item())
            emo_losses.append(emo_loss.item())

            # Accumulate per-video predictions
            sent_preds = sent_out.argmax(dim=1).cpu().tolist()
            emo_preds = emo_out.argmax(dim=1).cpu().tolist()

            for i, vid in enumerate(video_ids):
                video_sent_preds[vid].append(sent_preds[i])
                video_emo_preds[vid].append(emo_preds[i])
                video_sent_labels[vid] = sent_labels[i].item()
                video_emo_labels[vid] = emo_labels[i].item()

    # Video-level accuracy via majority vote
    agg_sent = aggregate_by_majority_vote(video_sent_preds)
    agg_emo = aggregate_by_majority_vote(video_emo_preds)

    vids = list(agg_sent.keys())
    sent_acc = accuracy_score([video_sent_labels[v] for v in vids],
                               [agg_sent[v] for v in vids])
    emo_acc = accuracy_score([video_emo_labels[v] for v in vids],
                              [agg_emo[v] for v in vids])

    return np.mean(sent_losses), np.mean(emo_losses), sent_acc, emo_acc


# ---------------------------
# MAIN TRAINING LOOP
# ---------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model.upper()}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Labels
    train_labels = build_label_dict(f"{MELD_BASE}/train/train_sent_emo.csv")
    val_labels = build_label_dict(f"{MELD_BASE}/dev_sent_emo.csv")

    # Datasets
    train_ds = FrameLevelMELDDataset(f"{FRAMES_BASE}/train", train_labels,
                                      transform=get_transform(training=True))
    val_ds = FrameLevelMELDDataset(f"{FRAMES_BASE}/val", val_labels,
                                   transform=get_transform(training=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Model
    if args.model == "cnn":
        model = MELD_CNN().to(device)
    else:
        model = MELD_VGG16(feature_dim=100).to(device)

    # Loss and optimizer
    sent_criterion = nn.CrossEntropyLoss().to(device)
    emo_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.9, 0.999), eps=1e-7)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {k: [] for k in
               ['train_sent_loss', 'train_emo_loss', 'val_sent_loss', 'val_emo_loss',
                'train_sent_acc', 'train_emo_acc', 'val_sent_acc', 'val_emo_acc']}

    for epoch in range(args.epochs):
        tr_sl, tr_el, tr_sa, tr_ea = run_epoch(
            model, train_loader, optimizer, sent_criterion, emo_criterion, device, training=True)
        vl_sl, vl_el, vl_sa, vl_ea = run_epoch(
            model, val_loader, optimizer, sent_criterion, emo_criterion, device, training=False)

        val_loss = vl_sl + vl_el

        for k, v in zip(history.keys(),
                        [tr_sl, tr_el, vl_sl, vl_el, tr_sa, tr_ea, vl_sa, vl_ea]):
            history[k].append(v)

        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train: sent_loss={tr_sl:.3f} emo_loss={tr_el:.3f} "
              f"sent_acc={tr_sa:.3f} emo_acc={tr_ea:.3f} | "
              f"Val: sent_loss={vl_sl:.3f} emo_loss={vl_el:.3f} "
              f"sent_acc={vl_sa:.3f} emo_acc={vl_ea:.3f}")

        # Early stopping + checkpointing
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(MODEL_DIR, f"best_{args.model}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Saved best model to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MELD visual stream model")
    parser.add_argument("--model", choices=["cnn", "vgg"], default="vgg")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()
    train(args)
