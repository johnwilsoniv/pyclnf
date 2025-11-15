#!/usr/bin/env python3
"""
PyCLNF + PyMTCNN Integration Example

Demonstrates best-practice integration with PyMTCNN face detector.
"""

import cv2
import sys


def main():
    print("PyCLNF + PyMTCNN Integration Example")
    print("=" * 60)

    # Check if PyMTCNN is installed
    try:
        from pymtcnn import PyMTCNN
    except ImportError:
        print("\n✗ PyMTCNN not installed!")
        print("  Install with: pip install pymtcnn")
        print("  Then run this example again.")
        sys.exit(1)

    from pyclnf import CLNF

    # Initialize detectors
    print("\n1. Initializing detectors...")
    mtcnn = PyMTCNN()  # Face detection
    clnf = CLNF(detector=None)  # Landmark refinement
    print("   ✓ PyMTCNN initialized")
    print("   ✓ CLNF initialized")

    # Load test image
    print("\n2. Loading image...")
    image_path = "test_image.jpg"  # Replace with your image

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        print(f"   ✓ Image loaded: {image.shape}")
    except Exception as e:
        print(f"\n✗ Error loading image: {e}")
        print("  Please provide a valid image path in the script")
        sys.exit(1)

    # Detect faces with MTCNN
    print("\n3. Detecting faces with PyMTCNN...")
    bboxes, landmarks_5 = mtcnn.detect(image, return_landmarks=True)

    if len(bboxes) == 0:
        print("   ✗ No faces detected!")
        sys.exit(1)

    print(f"   ✓ Detected {len(bboxes)} face(s)")

    # Process each face with CLNF
    print("\n4. Refining landmarks with CLNF...")

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        face_bbox = (x1, y1, x2 - x1, y2 - y1)  # Convert to [x, y, w, h]

        # Refine landmarks
        landmarks_68, info = clnf.fit(image, face_bbox)

        print(f"\n   Face {i + 1}:")
        print(f"     Bbox: {face_bbox}")
        print(f"     MTCNN: 5 landmarks")
        print(f"     CLNF: 68 landmarks")
        print(f"     Converged: {info['converged']}")
        print(f"     Iterations: {info['iterations']}")

        # Visualize results
        vis = image.copy()

        # Draw MTCNN bbox
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Draw MTCNN 5-point landmarks
        for x, y in landmarks_5[i]:
            cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Draw CLNF 68-point landmarks
        for x, y in landmarks_68:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

        # Save visualization
        output_path = f"pyclnf_pymtcnn_face{i + 1}.jpg"
        cv2.imwrite(output_path, vis)
        print(f"     Saved: {output_path}")

    print("\n" + "=" * 60)
    print("Integration example complete!")
    print("\nResults:")
    print("  - Blue box: MTCNN face detection")
    print("  - Red circles: MTCNN 5-point landmarks")
    print("  - Green circles: CLNF 68-point landmarks")


if __name__ == "__main__":
    main()
