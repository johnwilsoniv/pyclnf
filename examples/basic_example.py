#!/usr/bin/env python3
"""
Basic PyCLNF Example

Demonstrates basic usage of PyCLNF for facial landmark detection
with automatic face detection using built-in PyMTCNN.
"""

import cv2
from pyclnf import CLNF


def main():
    print("PyCLNF Basic Example")
    print("=" * 60)

    # Initialize CLNF with automatic face detection
    print("\n1. Initializing CLNF...")
    clnf = CLNF()  # Automatically includes PyMTCNN detector
    print("   ✓ CLNF initialized")

    # Load test image
    print("\n2. Loading test image...")
    image_path = "test_image.jpg"  # Replace with your image
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        print(f"   ✓ Image loaded: {image.shape}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("  Please provide a valid image path")
        return

    # Detect face and fit landmarks automatically
    print("\n3. Detecting face and fitting landmarks...")
    try:
        landmarks, info = clnf.detect_and_fit(image)

        print(f"   ✓ Detected {len(landmarks)} landmarks")
        print(f"   Converged: {info['converged']}")
        print(f"   Iterations: {info['iterations']}")
        print(f"   Bbox: {info['bbox']}")

        # Visualize landmarks
        print("\n4. Visualizing results...")
        vis = image.copy()

        # Draw landmarks
        for x, y in landmarks:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

        # Draw bounding box
        x, y, w, h = info['bbox']
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

        # Save visualization
        output_path = "pyclnf_output.jpg"
        cv2.imwrite(output_path, vis)
        print(f"   ✓ Saved to: {output_path}")

    except ValueError as e:
        print(f"\n✗ Error: {e}")
        return

    print("\n" + "=" * 60)
    print("Example complete!")
    print("\nResults:")
    print("  - Blue box: Face detected by PyMTCNN")
    print("  - Green circles: 68 CLNF landmarks")


if __name__ == "__main__":
    main()
