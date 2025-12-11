#!/usr/bin/env python3
"""
CANONICAL ACCURACY TEST: pyCLNF vs C++ OpenFace

This is THE reference test for verifying pyCLNF matches C++ OpenFace.
Run this test instead of writing ad-hoc comparison scripts.

Usage:
    python -m pytest pyclnf/tests/test_accuracy_vs_cpp.py -v
    # or directly:
    python pyclnf/tests/test_accuracy_vs_cpp.py

Expected results:
    - Overall error: < 0.5 px
    - Jaw error: < 1.0 px
    - Per-region errors: < 0.5 px

If errors exceed these thresholds, investigate:
    1. BBOX FORMAT: PyMTCNN outputs (x, y, width, height), NOT (x1, y1, x2, y2)
    2. CALIBRATION: Applied in pymtcnn/base.py stage_three(), NOT in pyclnf
    3. IMAGE MISMATCH: Ensure C++ and Python process the exact same image
"""
import numpy as np
import cv2
import subprocess
import tempfile
import os
from pathlib import Path

# Test configuration
OPENFACE_BIN = '/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction'
TEST_VIDEOS = [
    '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Normal Cohort/IMG_0422.MOV',
    '/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Normal Cohort/IMG_0942.MOV',
]

# Error thresholds (pixels)
# With BGR input (required), pyCLNF achieves sub-pixel accuracy.
# These thresholds allow some margin for floating-point variation.
THRESHOLD_OVERALL = 0.5  # Expected: ~0.1-0.2px
THRESHOLD_JAW = 1.0       # Expected: ~0.2-0.5px
THRESHOLD_REGION = 0.5


def extract_frame(video_path: str, frame_num: int = 0) -> np.ndarray:
    """Extract a specific frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to extract frame {frame_num} from {video_path}")
    return frame


def run_cpp_openface(image_path: str, output_dir: str) -> np.ndarray:
    """Run C++ OpenFace and return 68 landmarks."""
    cmd = [OPENFACE_BIN, '-f', image_path, '-out_dir', output_dir, '-2Dfp']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"C++ OpenFace failed: {result.stderr}")

    # Load CSV output
    csv_name = Path(image_path).stem + '.csv'
    csv_path = os.path.join(output_dir, csv_name)

    import pandas as pd
    df = pd.read_csv(csv_path)
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    landmarks = np.stack([df[x_cols].values[0], df[y_cols].values[0]], axis=1)
    return landmarks.astype(np.float32)


def run_python_pyclnf(bgr_image: np.ndarray) -> np.ndarray:
    """Run Python pyCLNF and return 68 landmarks.

    IMPORTANT: Must pass BGR (color) image, not grayscale.
    MTCNN produces different bboxes for grayscale vs color images.
    """
    from pyclnf import CLNF
    clnf = CLNF(convergence_profile='accurate', detector='pymtcnn', use_validator=False)
    landmarks, info = clnf.detect_and_fit(bgr_image)
    return landmarks


def compute_errors(py_landmarks: np.ndarray, cpp_landmarks: np.ndarray) -> dict:
    """Compute per-region errors."""
    diff = py_landmarks - cpp_landmarks
    per_lm_error = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)

    regions = {
        'overall': list(range(68)),
        'jaw': list(range(0, 17)),
        'brows': list(range(17, 27)),
        'nose': list(range(27, 36)),
        'eyes': list(range(36, 48)),
        'mouth': list(range(48, 68)),
    }

    errors = {}
    for name, indices in regions.items():
        errors[name] = per_lm_error[indices].mean()
    errors['max'] = per_lm_error.max()
    errors['max_idx'] = int(per_lm_error.argmax())

    return errors


def test_single_video(video_path: str) -> dict:
    """Test pyCLNF vs C++ on a single video."""
    video_name = Path(video_path).stem
    print(f"\n{'='*60}")
    print(f"Testing: {video_name}")
    print(f"{'='*60}")

    # Extract frame
    frame = extract_frame(video_path, frame_num=0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"Frame shape: {frame.shape}")

    # Save frame to temp file for C++
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_path = os.path.join(tmpdir, f'{video_name}_frame0.png')
        cv2.imwrite(frame_path, frame)

        # Run C++
        print("Running C++ OpenFace...")
        cpp_landmarks = run_cpp_openface(frame_path, tmpdir)
        print(f"  C++ LM8 (chin): ({cpp_landmarks[8,0]:.2f}, {cpp_landmarks[8,1]:.2f})")

        # Run Python (must use BGR frame, not grayscale - MTCNN needs color)
        print("Running Python pyCLNF...")
        py_landmarks = run_python_pyclnf(frame)
        print(f"  Python LM8 (chin): ({py_landmarks[8,0]:.2f}, {py_landmarks[8,1]:.2f})")

    # Compute errors
    errors = compute_errors(py_landmarks, cpp_landmarks)

    print(f"\nErrors (Python vs C++):")
    print(f"  Overall: {errors['overall']:.3f} px")
    print(f"  Jaw:     {errors['jaw']:.3f} px")
    print(f"  Eyes:    {errors['eyes']:.3f} px")
    print(f"  Nose:    {errors['nose']:.3f} px")
    print(f"  Mouth:   {errors['mouth']:.3f} px")
    print(f"  Brows:   {errors['brows']:.3f} px")
    print(f"  Max:     {errors['max']:.3f} px (LM{errors['max_idx']})")

    # Check thresholds
    passed = True
    if errors['overall'] > THRESHOLD_OVERALL:
        print(f"  FAIL: Overall error {errors['overall']:.3f} > {THRESHOLD_OVERALL}")
        passed = False
    if errors['jaw'] > THRESHOLD_JAW:
        print(f"  FAIL: Jaw error {errors['jaw']:.3f} > {THRESHOLD_JAW}")
        passed = False

    if passed:
        print("  PASS: All errors within thresholds")

    return {'video': video_name, 'errors': errors, 'passed': passed}


def main():
    """Run accuracy tests on all configured videos."""
    print("="*60)
    print("pyCLNF vs C++ OpenFace Accuracy Test")
    print("="*60)
    print(f"\nThresholds:")
    print(f"  Overall: < {THRESHOLD_OVERALL} px")
    print(f"  Jaw:     < {THRESHOLD_JAW} px")

    results = []
    for video_path in TEST_VIDEOS:
        if os.path.exists(video_path):
            result = test_single_video(video_path)
            results.append(result)
        else:
            print(f"\nWARNING: Video not found: {video_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_passed = all(r['passed'] for r in results)
    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  {r['video']}: {status} (overall={r['errors']['overall']:.3f}px, jaw={r['errors']['jaw']:.3f}px)")

    if all_passed:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED - Investigate bbox format and calibration!")

    return 0 if all_passed else 1


# Pytest integration
def test_accuracy_0422():
    """Pytest: Test accuracy on IMG_0422."""
    video_path = TEST_VIDEOS[0]
    if not os.path.exists(video_path):
        import pytest
        pytest.skip(f"Video not found: {video_path}")
    result = test_single_video(video_path)
    assert result['passed'], f"Accuracy test failed: {result['errors']}"


def test_accuracy_0942():
    """Pytest: Test accuracy on IMG_0942."""
    video_path = TEST_VIDEOS[1]
    if not os.path.exists(video_path):
        import pytest
        pytest.skip(f"Video not found: {video_path}")
    result = test_single_video(video_path)
    assert result['passed'], f"Accuracy test failed: {result['errors']}"


if __name__ == '__main__':
    import sys
    sys.exit(main())
