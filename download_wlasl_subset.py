"""
download_wlasl_subset.py
────────────────────────
Downloads a subset of the WLASL (Word-Level American Sign Language) dataset.

Requirements:
    pip install requests tqdm

Usage:
    1. Place WLASL_v0.3.json in the same directory as this script
       (or update METADATA_FILE path below).
    2. Run:  python download_wlasl_subset.py

Output folder structure:
    wlasl/videos/<WORD>/<video_id>.mp4

    Example:
        wlasl/videos/HELLO/hello_001.mp4
        wlasl/videos/HELLO/hello_002.mp4
"""

import os
import json
import requests
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

# Path to the WLASL metadata JSON file
# Place WLASL_v0.3.json inside the wlasl/ folder (or update this path)
METADATA_FILE = os.path.join("wlasl", "WLASL_v0.3.json")

# Root output directory
OUTPUT_DIR = os.path.join("wlasl", "videos")

# Maximum videos to download per word (keeps dataset manageable)
MAX_VIDEOS_PER_WORD = 50

# Request timeout in seconds per video download
DOWNLOAD_TIMEOUT = 30

# Words to download (40 practical ASL vocabulary words)
TARGET_WORDS = [
    # Greetings / politeness
    "hello", "thank_you", "please", "sorry", "yes", "no",
    "stop", "go", "come", "wait",

    # Daily activities
    "help", "eat", "drink", "sleep", "work",
    "study", "read", "write", "learn", "teach",

    # Emotions / states
    "good", "bad", "happy", "sad", "love",
    "like", "want", "need", "know", "think",

    # Actions / objects
    "buy", "pay", "open", "close", "give",
    "take", "show", "look", "listen", "play",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_metadata(path: str) -> list:
    """Load and return the WLASL JSON metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Metadata file not found: '{path}'\n"
            "Download it from: https://github.com/dxli94/WLASL"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_word_index(metadata: list, target_words: list) -> dict:
    """
    Build a lookup: { word_lower: [instance_dict, ...] }
    Only for words in target_words.
    """
    target_set = {w.lower() for w in target_words}
    index = {}
    for entry in metadata:
        word = entry.get("gloss", "").lower()
        if word in target_set:
            index[word] = entry.get("instances", [])
    return index


def download_video(url: str, save_path: str, timeout: int = DOWNLOAD_TIMEOUT) -> bool:
    """
    Download a single video from url and save to save_path.
    Returns True on success, False on any error.
    """
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True

    except requests.exceptions.RequestException as e:
        # Remove partially downloaded file if it exists
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  WLASL Subset Downloader")
    print("=" * 60)

    # 1. Load metadata
    print(f"\n📂  Loading metadata from '{METADATA_FILE}' …")
    metadata = load_metadata(METADATA_FILE)
    print(f"    Loaded {len(metadata)} glosses from dataset.")

    # 2. Build per-word index
    word_index = build_word_index(metadata, TARGET_WORDS)

    found_words    = sorted(word_index.keys())
    missing_words  = sorted(set(w.lower() for w in TARGET_WORDS) - set(found_words))

    print(f"\n✅  Found  : {len(found_words)}  words — {found_words}")
    if missing_words:
        print(f"⚠️   Missing: {len(missing_words)} words — {missing_words}")

    # 3. Download videos word by word
    total_downloaded = 0
    total_skipped    = 0

    for word in found_words:
        instances = word_index[word]

        # Folder: wlasl/videos/HELLO/
        word_folder = os.path.join(OUTPUT_DIR, word.upper())
        os.makedirs(word_folder, exist_ok=True)

        # Limit to MAX_VIDEOS_PER_WORD
        instances_to_download = instances[:MAX_VIDEOS_PER_WORD]

        print(f"\n{'─'*60}")
        print(f"📹  Word: {word.upper()}  "
              f"({len(instances_to_download)} of {len(instances)} instances)")

        word_downloaded = 0
        word_skipped    = 0

        for instance in tqdm(instances_to_download, desc=f"  {word.upper()}", unit="video"):
            video_id = instance.get("video_id", "")
            url      = instance.get("url", "")

            if not url or not video_id:
                word_skipped += 1
                continue

            # Build filename: hello_001.mp4 (word prefix + zero-padded id)
            filename  = f"{word}_{str(video_id).zfill(3)}.mp4"
            save_path = os.path.join(word_folder, filename)

            # Skip if already downloaded
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                word_downloaded += 1
                continue

            # Download
            success = download_video(url, save_path)
            if success:
                word_downloaded += 1
            else:
                word_skipped += 1

        total_downloaded += word_downloaded
        total_skipped    += word_skipped

        print(f"  ✔  Downloaded: {word_downloaded}  |  Skipped/Failed: {word_skipped}")

    # 4. Summary
    print(f"\n{'='*60}")
    print(f"  Download complete!")
    print(f"  Total videos downloaded : {total_downloaded}")
    print(f"  Total skipped / failed  : {total_skipped}")
    print(f"  Saved to                : {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*60}\n")
    print("  Next steps:")
    print("  1. Collect landmarks: python landmark_collection.py --label HELLO")
    print("     (or use the downloaded videos with a video-landmark extractor)")
    print("  2. Retrain           : python train_model.py")
    print("  3. Run app           : python app.py\n")


if __name__ == "__main__":
    main()
