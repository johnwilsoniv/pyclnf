#!/usr/bin/env python3
"""
PyCLNF with Built-in PyMTCNN Example

Demonstrates automatic face detection with CLNF landmark refinement.
"""

import cv2
import sys


def main():
    print("PyCLNF with PyMTCNN Example")
    print("=" * 60)

    from pyclnf import CLNF

    # Initialize CLNF with built-in PyMTCNN detector
    print("\n1. Initializing CLNF...")
    clnf = CLNF()  # Automatically initializes PyMTCNN
    print("   [OK] CLNF initialized with PyMTCNN detector")

    # Load test image
    print("\n2. Loading image...")
    image_path = "test_image.jpg"  # Replace with your image

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        print(f"   [OK] Image loaded: {image.shape}")
    except Exception as e:
        print(f"\n[FAILED] Error loading image: {e}")
        print("  Please provide a valid image path in the script")
        sys.exit(1)

    # Detect faces and fit landmarks automatically
    print("\n3. Detecting faces and fitting landmarks...")
    try:
        landmarks_68, info = clnf.detect_and_fit(image)

        print(f"\n   [OK] Success!")
        print(f"     Detected 68 landmarks")
        print(f"     Converged: {info['converged']}")
        print(f"     Iterations: {info['iterations']}")
        print(f"     Bbox: {info['bbox']}")

        # Visualize results
        vis = image.copy()

        # Draw bbox
        x, y, w, h = info['bbox']
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

        # Draw CLNF 68-point landmarks
        for lx, ly in landmarks_68:
            cv2.circle(vis, (int(lx), int(ly)), 2, (0, 255, 0), -1)

        # Save visualization
        output_path = "pyclnf_output.jpg"
        cv2.imwrite(output_path, vis)
        print(f"\n   Saved: {output_path}")

    except ValueError as e:
        print(f"\n   [FAILED] Error: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Example complete!")
    print("\nResults:")
    print("  - Blue box: PyMTCNN face detection (built-in)")
    print("  - Green circles: CLNF 68-point landmarks")


if __name__ == "__main__":
    main()
