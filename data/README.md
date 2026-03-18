# Data

Raw data is **not committed** to this repository.

## MELD Dataset

**Multimodal EmotionLines Dataset** — multi-party conversations from the *Friends* TV series, annotated with emotion and sentiment labels at utterance level.

**Download options:**
- Kaggle: https://www.kaggle.com/datasets/zaber666/meld-dataset
- Official: https://affective-meld.github.io/

## Expected Directory Layout

```
data/
└── MELD-RAW/
    └── MELD.Raw/
        ├── train/
        │   ├── train_sent_emo.csv
        │   └── train_splits/          # .mp4 video clips
        ├── dev/
        │   ├── dev_sent_emo.csv
        │   └── dev_splits_complete/
        └── test/
            ├── test_sent_emo.csv
            └── output_repeated_splits_test/
```

Update the paths in the notebooks or `src/dataset.py` to match your local layout.

## Frame Extraction

The notebooks extract one frame every 3 seconds from each clip. Extracted frames are saved as:

```
frames/
├── train/<dialogue_id>_<utterance_id>/frame_0.jpg, frame_1.jpg, ...
├── val/...
└── test/...
```
