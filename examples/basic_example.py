#!/usr/bin/env python3
"""
Basic PyCLNF Example

Demonstrates basic usage of PyCLNF for facial landmark detection.
"""

import cv2
import numpy as np
from pyclnf import CLNF


def main():
    print("PyCLNF Basic Example")
    print("=" * 60)

    # Initialize CLNF detector (no built-in face detector)
    print("\n1. Initializing CLNF detector...")
    clnf = CLNF(detector=None)
    print("   ✓ CLNF initialized")

    # Create a test image (replace with your image)
    print("\n2. Loading test image...")
    # For this example, we create a blank image
    # In practice, load with: image = cv2.imread("your_image.jpg")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    print(f"   ✓ Image loaded: {image.shape}")

    # Define face bounding box [x, y, width, height]
    # In practice, get this from a face detector (MTCNN, RetinaFace, etc.)
    face_bbox = (200, 120, 200, 250)
    print(f"\n3. Face bbox: {face_bbox}")

    # Detect landmarks
    print("\n4. Detecting landmarks...")
    landmarks, info = clnf.fit(image, face_bbox)

    print(f"   ✓ Detected {len(landmarks)} landmarks")
    print(f"   Converged: {info['converged']}")
    print(f"   Iterations: {info['iterations']}")
    print(f"   Final update: {info['final_update']:.6f}")

    # Visualize landmarks
    print("\n5. Visualizing results...")
    vis = image.copy()
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

    # Draw bounding box
    x, y, w, h = face_bbox
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Save visualization
    output_path = "pyclnf_basic_example.jpg"
    cv2.imwrite(output_path, vis)
    print(f"   ✓ Saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("\nNext steps:")
    print("  - Use with a real image and face detector")
    print("  - Try PyMTCNN for face detection")
    print("  - Explore multi-scale refinement options")


if __name__ == "__main__":
    main()
