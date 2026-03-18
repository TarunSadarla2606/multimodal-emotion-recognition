"""
dataset.py
Frame extraction and PyTorch Dataset for the MELD visual stream.

Pipeline:
  1. extract_frames()         — sample one frame every N seconds from each .mp4 clip
  2. build_label_dict()       — map video filenames to (emotion, sentiment) labels
  3. FrameLevelMELDDataset    — PyTorch Dataset returning (frame_tensor, labels, video_id)
"""

import os
import glob
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from models import EMOTION2IDX, SENTIMENT2IDX


# ---------------------------
# FRAME EXTRACTION
# ---------------------------

def extract_frames(video_dir, output_dir, csv_path, frame_interval_sec=3):
    """
    Extract one frame every `frame_interval_sec` seconds from each MELD video clip.

    Args:
        video_dir (str)        : Directory containing .mp4 utterance clips
        output_dir (str)       : Root output directory for extracted frames
        csv_path (str)         : MELD CSV (train/val/test) for valid utterance IDs
        frame_interval_sec (int): Seconds between sampled frames (default: 3)

    Output structure:
        output_dir/<dialogue_id>_<utterance_id>/frame_0.jpg, frame_1.jpg, ...
    """
    df = pd.read_csv(csv_path)
    valid_videos = set(
        f"{row['Dialogue_ID']}_{row['Utterance_ID']}.mp4"
        for _, row in df.iterrows()
    )

    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    skipped, extracted = 0, 0
    for video_file in tqdm(video_files, desc=f"Extracting frames from {os.path.basename(video_dir)}"):
        if video_file not in valid_videos:
            skipped += 1
            continue

        video_path = os.path.join(video_dir, video_file)
        video_key = video_file.replace('.mp4', '')
        frame_out_dir = os.path.join(output_dir, video_key)
        os.makedirs(frame_out_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            continue

        frame_step = max(1, int(fps * frame_interval_sec))
        frame_idx, saved = 0, 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                out_path = os.path.join(frame_out_dir, f"frame_{saved}.jpg")
                cv2.imwrite(out_path, frame)
                saved += 1
            frame_idx += 1

        cap.release()
        if saved > 0:
            extracted += 1

    print(f"Done. Extracted frames from {extracted} videos. Skipped {skipped}.")


def extract_all_splits(base_dir, output_base="frames"):
    """
    Convenience function to extract frames from all three MELD splits.

    Args:
        base_dir (str)   : Root of MELD-RAW/MELD.Raw/
        output_base (str): Root output directory for all frame splits
    """
    splits = {
        "train": {
            "csv": os.path.join(base_dir, "train/train_sent_emo.csv"),
            "video_dir": os.path.join(base_dir, "train/train_splits"),
            "output_dir": os.path.join(output_base, "train"),
        },
        "val": {
            "csv": os.path.join(base_dir, "dev_sent_emo.csv"),
            "video_dir": os.path.join(base_dir, "dev/dev_splits_complete"),
            "output_dir": os.path.join(output_base, "val"),
        },
        "test": {
            "csv": os.path.join(base_dir, "test/test_sent_emo.csv"),
            "video_dir": os.path.join(base_dir, "test/output_repeated_splits_test"),
            "output_dir": os.path.join(output_base, "test"),
        },
    }
    for split, cfg in splits.items():
        print(f"\n--- {split.upper()} ---")
        extract_frames(cfg["video_dir"], cfg["output_dir"], cfg["csv"])


# ---------------------------
# LABEL BUILDING
# ---------------------------

def build_label_dict(csv_path):
    """
    Build a mapping from video filename → {emotion: int, sentiment: int}.

    Args:
        csv_path (str): Path to MELD train/val/test CSV

    Returns:
        dict: {'{dialogue_id}_{utterance_id}.mp4': {'emotion': int, 'sentiment': int}}
    """
    df = pd.read_csv(csv_path)
    labels = {}

    for _, row in df.iterrows():
        try:
            key = f"{row['Dialogue_ID']}_{row['Utterance_ID']}.mp4"
            labels[key] = {
                'emotion': EMOTION2IDX[row['Emotion'].lower()],
                'sentiment': SENTIMENT2IDX[row['Sentiment'].lower()],
            }
        except (KeyError, AttributeError) as e:
            print(f"[WARN] Skipping row — {e}: {row.get('Dialogue_ID')}, {row.get('Utterance_ID')}")

    print(f"[INFO] Built label dict: {len(labels)} entries from {csv_path}")
    return labels


# ---------------------------
# PYTORCH DATASET
# ---------------------------

class FrameLevelMELDDataset(Dataset):
    """
    PyTorch Dataset for frame-level visual classification on MELD.

    Each sample is one frame from a video clip, paired with the utterance-level
    emotion and sentiment labels (same label for all frames of the same clip).

    Args:
        frames_root (str)  : Directory containing per-video subdirectories of frames
        label_dict (dict)  : Output of build_label_dict()
        transform          : torchvision transform applied to each frame

    Returns per item:
        image (Tensor)     : (3, H, W) normalized frame
        sentiment (int)    : sentiment class index
        emotion (int)      : emotion class index
        video_folder (str) : video identifier (for video-level aggregation)
    """

    def __init__(self, frames_root, label_dict, transform=None):
        self.samples = []
        self.transform = transform

        for video_folder in os.listdir(frames_root):
            video_path = os.path.join(frames_root, video_folder)
            if not os.path.isdir(video_path):
                continue

            video_key = video_folder + ".mp4"
            if video_key not in label_dict:
                continue

            label = label_dict[video_key]
            frame_list = sorted(glob.glob(os.path.join(video_path, 'frame_*.jpg')))
            if not frame_list:
                continue

            for frame_path in frame_list:
                self.samples.append((
                    frame_path,
                    label['sentiment'],
                    label['emotion'],
                    video_folder,
                ))

        print(f"[INFO] Dataset: {len(self.samples)} frames from {frames_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, sentiment, emotion, video_folder = self.samples[idx]
        image = Image.open(frame_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, (sentiment, emotion, video_folder)


# ---------------------------
# STANDARD TRANSFORM
# ---------------------------

def get_transform(training=False):
    """
    Return ImageNet-normalized transform.
    Training adds random horizontal flip for augmentation.
    """
    ops = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    if training:
        ops.insert(0, transforms.RandomHorizontalFlip())
    return transforms.Compose(ops)


if __name__ == "__main__":
    # Quick sanity check
    labels = build_label_dict("data/MELD-RAW/MELD.Raw/train/train_sent_emo.csv")
    dataset = FrameLevelMELDDataset("frames/train", labels, transform=get_transform())
    print(f"Dataset size: {len(dataset)}")
    img, (sent, emo, vid) = dataset[0]
    print(f"Frame shape: {img.shape} | Sentiment: {sent} | Emotion: {emo} | Video: {vid}")
