"""
augment_sequences.py
─────────────────────
Multiplies the hand-landmark .npy sequences in sequence_dataset/ using
four lightweight augmentation strategies so the LSTM has more data to
learn from despite the high number of corrupt WLASL source videos.

Augmentations applied per original sample
──────────────────────────────────────────
  1. Gaussian jitter   – adds tiny random noise to each landmark coordinate
  2. Mirror (flip)     – negates all x-coordinates (mirrors the hand)
  3. Scale             – randomly scales landmark magnitudes ±20 %
  4. Time-warp         – slightly stretches / compresses the temporal axis

Each augmentation produces one new .npy of the same shape (SEQUENCE_LENGTH, 42).
So every original sample yields 4 augmented copies → ~5× total dataset size.

Output: files are saved alongside originals as
        sample_<N>_jitter.npy / _mirror.npy / _scale.npy / _timewarp.npy
"""

import os
import numpy as np
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

SEQUENCE_ROOT   = "sequence_dataset"
SEQUENCE_LENGTH = 30   # must match extract_wlasl_landmarks.py

JITTER_STD  = 0.005   # std-dev of Gaussian noise (in normalised landmark units)
SCALE_RANGE = (0.80, 1.20)   # uniform scale multiplier range

# ── Augmentation functions ─────────────────────────────────────────────────────

def augment_jitter(seq: np.ndarray) -> np.ndarray:
    """Add small Gaussian noise independently to every coordinate."""
    noise = np.random.normal(0, JITTER_STD, seq.shape).astype(np.float32)
    return seq + noise


def augment_mirror(seq: np.ndarray) -> np.ndarray:
    """Mirror the hand by negating all x-coordinates (indices 0, 2, 4, …)."""
    mirrored = seq.copy()
    mirrored[:, 0::2] *= -1   # x channels are even indices in the 42-vector
    return mirrored


def augment_scale(seq: np.ndarray) -> np.ndarray:
    """Randomly scale landmark magnitudes (simulates different hand sizes / distances)."""
    factor = np.random.uniform(*SCALE_RANGE)
    return (seq * factor).astype(np.float32)


def augment_timewarp(seq: np.ndarray) -> np.ndarray:
    """
    Slight temporal warp: randomly select SEQUENCE_LENGTH indices from a
    slightly stretched or compressed version of the sequence and re-sample.
    This simulates signing at different speeds.
    """
    T = seq.shape[0]
    # warp factor: 0.8–1.2 of normal speed
    warp = np.random.uniform(0.8, 1.2)
    src_len  = max(int(T * warp), 2)
    src_idx  = np.linspace(0, T - 1, src_len)

    # Build warped sequence via linear interpolation
    warped = np.stack([
        (
            seq[int(i)] * (1 - (i % 1)) +
            seq[min(int(i) + 1, T - 1)] * (i % 1)
        ).astype(np.float32)
        for i in src_idx
    ])

    # Re-sample back to SEQUENCE_LENGTH
    target_idx = np.linspace(0, len(warped) - 1, SEQUENCE_LENGTH, dtype=int)
    return warped[target_idx]


AUGMENTATIONS = {
    "jitter":   augment_jitter,
    "mirror":   augment_mirror,
    "scale":    augment_scale,
    "timewarp": augment_timewarp,
}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Sequence Augmentor")
    print("=" * 60)

    if not os.path.exists(SEQUENCE_ROOT):
        print(f"\n❌  Sequence root not found: '{SEQUENCE_ROOT}'")
        print("    Run extract_wlasl_landmarks.py first.\n")
        return

    word_folders = sorted([
        d for d in os.listdir(SEQUENCE_ROOT)
        if os.path.isdir(os.path.join(SEQUENCE_ROOT, d))
    ])

    if not word_folders:
        print(f"\n⚠️   No word folders found inside '{SEQUENCE_ROOT}'.\n")
        return

    total_original  = 0
    total_augmented = 0
    skipped_aug     = 0   # already-augmented files we won't re-augment

    for word in tqdm(word_folders, desc="Words", unit="word"):
        word_dir = os.path.join(SEQUENCE_ROOT, word)

        # Only augment original samples (sample_0.npy, sample_1.npy, …)
        originals = sorted([
            f for f in os.listdir(word_dir)
            if f.startswith("sample_") and
               not any(tag in f for tag in AUGMENTATIONS) and
               f.endswith(".npy")
        ])

        total_original += len(originals)

        for fname in originals:
            src_path = os.path.join(word_dir, fname)
            seq = np.load(src_path)   # shape: (30, 42)

            stem = fname[:-4]  # e.g. "sample_0"

            for aug_name, aug_fn in AUGMENTATIONS.items():
                out_name = f"{stem}_{aug_name}.npy"
                out_path = os.path.join(word_dir, out_name)

                if os.path.exists(out_path):
                    skipped_aug += 1
                    continue   # already exists from a previous run

                aug_seq = aug_fn(seq)
                assert aug_seq.shape == (SEQUENCE_LENGTH, 42), \
                    f"Unexpected shape after {aug_name}: {aug_seq.shape}"
                np.save(out_path, aug_seq)
                total_augmented += 1

    print(f"\n{'='*60}")
    print(f"  Augmentation complete!")
    print(f"  Original sequences      : {total_original}")
    print(f"  New augmented sequences : {total_augmented}")
    print(f"  Already existed (skip)  : {skipped_aug}")
    print(f"  Total sequences now     : {total_original + total_augmented + skipped_aug}")
    print(f"  Approx. samples/word    : {(total_original + total_augmented + skipped_aug) // max(len(word_folders),1)}")
    print(f"{'='*60}")
    print(f"\n  Next step: train the LSTM model")
    print(f"    python train_model.py\n")


if __name__ == "__main__":
    main()
