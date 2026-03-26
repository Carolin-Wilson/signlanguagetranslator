"""
extract_wlasl_landmarks.py
──────────────────────────
Converts WLASL .mp4 videos into fixed-length landmark sequences for LSTM training.

For each video:
  1. Read all frames with detected hand landmarks.
  2. Evenly sample exactly SEQUENCE_LENGTH frames from that set.
  3. Extract a 42-feature vector (21 landmarks × x, y) per frame, normalised
     relative to the wrist (landmark 0) so the sequence is position-invariant.
  4. Save as a .npy file of shape (SEQUENCE_LENGTH, 42).

Input structure:   wlasl/videos/<WORD>/*.mp4
Output structure:  sequence_dataset/<WORD>/sample_<N>.npy

Requirements:
    pip install opencv-python mediapipe numpy tqdm
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

# Root folder containing per-word video subfolders
VIDEO_ROOT = os.path.join("wlasl", "videos")

# Root folder for output .npy sequences
OUTPUT_ROOT = "sequence_dataset"

# Number of frames per sequence (temporal window for LSTM)
SEQUENCE_LENGTH = 30

# Minimum detected-hand frames required to keep a video.
# Videos with fewer detected frames than this are skipped entirely.
# Set low so that even short clips can be interpolated up to SEQUENCE_LENGTH.
MIN_HAND_FRAMES = 5

# MediaPipe confidence thresholds — lower = detects more hands in tricky lighting
DETECTION_CONFIDENCE  = 0.3
TRACKING_CONFIDENCE   = 0.3

# ── MediaPipe setup ───────────────────────────────────────────────────────────

mp_hands   = mp.solutions.hands
hand_model = mp_hands.Hands(
    static_image_mode=False,       # video mode: uses tracking between frames → detects more
    max_num_hands=1,
    min_detection_confidence=DETECTION_CONFIDENCE,
    min_tracking_confidence=TRACKING_CONFIDENCE,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_landmark_vector(hand_landmarks) -> np.ndarray:
    """
    Convert a MediaPipe hand_landmarks object into a flat 42-element numpy array.
    Coordinates are normalised relative to the wrist (landmark index 0) so the
    feature vector is invariant to hand position in the frame.

    Returns: np.ndarray of shape (42,) — [x0, y0, x1, y1, ..., x20, y20]
    """
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y

    vector = []
    for lm in hand_landmarks.landmark:
        vector.append(lm.x - wrist_x)
        vector.append(lm.y - wrist_y)

    return np.array(vector, dtype=np.float32)  # shape: (42,)


def extract_sequence_from_video(video_path: str) -> np.ndarray | None:
    """
    Process a single video file and return a landmark sequence of shape
    (SEQUENCE_LENGTH, 42), or None if the video should be skipped.

    Strategy:
      - Collect all frames that have a detected hand.
      - If fewer than MIN_HAND_FRAMES detected → skip.
      - Evenly sample SEQUENCE_LENGTH indices from the detected-frame list.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None  # corrupted / unreadable file

    detected_frames = []  # list of (42,) landmark vectors

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_model.process(rgb)

        if results.multi_hand_landmarks:
            # Take the first detected hand
            vector = extract_landmark_vector(results.multi_hand_landmarks[0])
            detected_frames.append(vector)

    cap.release()

    # Not enough hand-detected frames → skip this video
    if len(detected_frames) < MIN_HAND_FRAMES:
        return None

    detected_arr = np.stack(detected_frames)  # shape: (N, 42)

    if len(detected_frames) >= SEQUENCE_LENGTH:
        # Enough frames: evenly sub-sample down to SEQUENCE_LENGTH
        indices  = np.linspace(0, len(detected_frames) - 1, SEQUENCE_LENGTH, dtype=int)
        sequence = detected_arr[indices]
    else:
        # Fewer frames than needed: linearly interpolate up to SEQUENCE_LENGTH
        # This preserves the motion shape rather than just repeating frames.
        src_idx  = np.linspace(0, len(detected_frames) - 1, SEQUENCE_LENGTH)
        sequence = np.stack([
            (
                detected_arr[int(i)] * (1 - (i % 1)) +
                detected_arr[min(int(i) + 1, len(detected_frames) - 1)] * (i % 1)
            ).astype(np.float32)
            for i in src_idx
        ])

    return sequence  # shape: (SEQUENCE_LENGTH, 42)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  WLASL Landmark Extractor → LSTM Sequence Dataset")
    print("=" * 60)

    if not os.path.exists(VIDEO_ROOT):
        print(f"\n❌  Video root not found: '{VIDEO_ROOT}'")
        print("    Run download_wlasl_subset.py first.\n")
        return

    # Discover all word folders
    word_folders = sorted([
        d for d in os.listdir(VIDEO_ROOT)
        if os.path.isdir(os.path.join(VIDEO_ROOT, d))
    ])

    if not word_folders:
        print(f"\n⚠️   No word folders found inside '{VIDEO_ROOT}'.\n")
        return

    print(f"\n  Words found : {len(word_folders)}")
    print(f"  Output root : {os.path.abspath(OUTPUT_ROOT)}\n")

    total_saved   = 0
    total_skipped = 0

    for word in word_folders:
        video_dir  = os.path.join(VIDEO_ROOT, word)
        output_dir = os.path.join(OUTPUT_ROOT, word)
        os.makedirs(output_dir, exist_ok=True)

        # Collect all .mp4 files for this word
        video_files = sorted([
            f for f in os.listdir(video_dir)
            if f.lower().endswith(".mp4")
        ])

        if not video_files:
            print(f"  ⚠️  No .mp4 files for '{word}' — skipping.")
            continue

        print(f"\n{'─'*60}")
        print(f"  Word: {word}  ({len(video_files)} videos)")

        word_saved   = 0
        word_skipped = 0

        for video_file in tqdm(video_files, desc=f"  {word}", unit="video"):
            video_path = os.path.join(video_dir, video_file)

            try:
                sequence = extract_sequence_from_video(video_path)
            except Exception:
                # Any unexpected error (corrupt codec, etc.) → skip silently
                word_skipped += 1
                continue

            if sequence is None:
                # Not enough hand-detected frames
                word_skipped += 1
                continue

            # Verify shape before saving (defensive check)
            assert sequence.shape == (SEQUENCE_LENGTH, 42), \
                f"Unexpected shape {sequence.shape}"

            # Save as sample_0.npy, sample_1.npy, …
            out_filename = f"sample_{word_saved}.npy"
            out_path     = os.path.join(output_dir, out_filename)
            np.save(out_path, sequence)
            word_saved += 1

        total_saved   += word_saved
        total_skipped += word_skipped

        print(f"  ✔  Saved: {word_saved}  |  Skipped: {word_skipped}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Extraction complete!")
    print(f"  Total sequences saved   : {total_saved}")
    print(f"  Total videos skipped    : {total_skipped}")
    print(f"  Output directory        : {os.path.abspath(OUTPUT_ROOT)}")
    print(f"  Each .npy shape         : ({SEQUENCE_LENGTH}, 42)")
    print(f"{'='*60}")
    print(f"\n  Next step: train the LSTM model")
    print(f"    python train_lstm_model.py\n")


if __name__ == "__main__":
    main()
