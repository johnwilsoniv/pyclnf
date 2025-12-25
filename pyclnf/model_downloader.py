"""
Model downloader for pyclnf.

Downloads model files from GitHub releases on first use.
This allows the PyPI package to stay under size limits while
providing full model access.
"""

import os
import sys
import hashlib
import urllib.request
from pathlib import Path

# GitHub release URL for model files
GITHUB_RELEASE_URL = "https://github.com/johnwilsoniv/pyclnf/releases/download/v0.2.5-models"

# Model files and their SHA256 hashes
MODEL_FILES = {
    "patch_experts/cen_patches_0.25_of.dat": {
        "size": 60602360,
        "sha256": None,  # Will be set after upload
    },
    "patch_experts/cen_patches_0.35_of.dat": {
        "size": 60602360,
        "sha256": None,
    },
    "patch_experts/cen_patches_0.50_of.dat": {
        "size": 154289792,
        "sha256": None,
    },
    "patch_experts/cen_patches_1.00_of.dat": {
        "size": 154289792,
        "sha256": None,
    },
}


def get_models_dir():
    """Get the models directory path."""
    return Path(__file__).parent / "models"


def models_exist():
    """Check if all model files exist."""
    models_dir = get_models_dir()
    for rel_path in MODEL_FILES.keys():
        if not (models_dir / rel_path).exists():
            return False
    return True


def download_file(url, dest_path, expected_size=None, show_progress=True):
    """Download a file with progress indication."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            if expected_size and total_size != expected_size:
                print(f"Warning: Expected {expected_size} bytes, got {total_size}")

            downloaded = 0
            chunk_size = 8192

            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if show_progress and total_size > 0:
                        pct = downloaded * 100 // total_size
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r  {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)")
                        sys.stdout.flush()

            if show_progress:
                print()  # Newline after progress

        return True

    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def download_models(force=False):
    """
    Download model files from GitHub releases if not present.

    Args:
        force: Re-download even if files exist

    Returns:
        bool: True if all models are available
    """
    models_dir = get_models_dir()

    if not force and models_exist():
        return True

    print("Downloading pyclnf model files (~410 MB)...")
    print(f"Destination: {models_dir}")
    print()

    success = True
    for rel_path, info in MODEL_FILES.items():
        dest_path = models_dir / rel_path

        if not force and dest_path.exists():
            print(f"  ✓ {rel_path} (already exists)")
            continue

        # Use just the filename for the URL (GitHub release strips directory)
        filename = Path(rel_path).name
        url = f"{GITHUB_RELEASE_URL}/{filename}"
        print(f"  Downloading {rel_path}...")

        if not download_file(url, dest_path, info.get("size")):
            success = False
            print(f"  ✗ Failed to download {rel_path}")
        else:
            print(f"  ✓ {rel_path}")

    if success:
        print()
        print("All models downloaded successfully!")
    else:
        print()
        print("Some models failed to download. Try again or download manually from:")
        print(f"  {GITHUB_RELEASE_URL}")

    return success


def ensure_models():
    """Ensure models are available, downloading if necessary."""
    if not models_exist():
        return download_models()
    return True


if __name__ == "__main__":
    # Allow running as: python -m pyclnf.model_downloader
    import argparse
    parser = argparse.ArgumentParser(description="Download pyclnf model files")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    success = download_models(force=args.force)
    sys.exit(0 if success else 1)
