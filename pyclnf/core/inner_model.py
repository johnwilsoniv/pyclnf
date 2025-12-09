"""
Hierarchical Inner Face Model for refining landmarks 17-67.

This implements the OpenFace hierarchical inner model refinement using:
- 51-point Inner PDM (landmarks 17-67 mapped to 0-50)
- CCNF patch experts (older format than CEN)
- NU-RLMS optimization with C++ parameters (reg=2.5, sigma=1.75, ws=9)

C++ Reference: OpenFace LandmarkDetectorModel.cpp lines 643-800
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .inner_pdm import InnerPDM
from .ccnf_patch_expert import CCNFPatchExperts
from .inner_optimizer import InnerModelOptimizer


class HierarchicalInnerModel:
    """
    Full hierarchical inner face model refinement.

    Refines inner face landmarks (eyebrows, nose, eyes, mouth) using
    a dedicated 51-point PDM and CCNF patch experts.

    C++ OpenFace refinement sequence (LandmarkDetectorModel.cpp:643-800):
    1. Extract inner landmarks from main model (17-67 -> 0-50)
    2. Fit inner PDM to extracted landmarks (CalcParams)
    3. Check scale requirement (main_scale > 0.9 * inner_patch_scale)
    4. Run full NU-RLMS optimization (DetectLandmarks)
    5. Map refined landmarks back to main model
    """

    # Mapping from main 68-point model to inner 51-point model
    # Main landmarks 17-67 map to inner landmarks 0-50
    MAIN_TO_INNER = {i: i - 17 for i in range(17, 68)}
    INNER_TO_MAIN = {i - 17: i for i in range(17, 68)}

    def __init__(self, model_dir: str):
        """
        Initialize hierarchical inner model.

        Args:
            model_dir: Base directory containing exported models
                       (expects exported_inner_pdm/ and exported_inner_ccnf/)
        """
        model_dir = Path(model_dir)

        # Load inner PDM (51 points)
        pdm_dir = model_dir / 'exported_inner_pdm'
        if pdm_dir.exists():
            self.pdm = InnerPDM(str(pdm_dir))
            print(f"  Inner PDM loaded (51 points, 32 modes)")
        else:
            raise FileNotFoundError(f"Inner PDM not found at {pdm_dir}")

        # Load CCNF patch experts
        ccnf_dir = model_dir / 'exported_inner_ccnf'
        if ccnf_dir.exists():
            self.ccnf_experts = CCNFPatchExperts(str(ccnf_dir))
        else:
            raise FileNotFoundError(f"Inner CCNF not found at {ccnf_dir}")

        # Create optimizer with C++ inner model parameters
        self.optimizer = InnerModelOptimizer()

        # Inner model is trained at scale 1.0
        self.patch_scaling = 1.0
        self.min_scale_ratio = 0.9  # C++ threshold

    def refine_landmarks(self,
                         image: np.ndarray,
                         main_landmarks: np.ndarray,
                         main_params: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Refine inner face landmarks using full NU-RLMS optimization.

        C++ algorithm (LandmarkDetectorModel.cpp:643-800):
        1. Extract inner landmarks from main (17-67 -> 0-50)
        2. Fit inner PDM to extracted landmarks
        3. Check scale requirement
        4. Run NU-RLMS optimization
        5. Map refined landmarks back to main

        Args:
            image: Grayscale input image
            main_landmarks: Full 68-point landmarks from main model
            main_params: Optional main model PDM parameters (for scale check)

        Returns:
            refined_landmarks: Full 68-point landmarks with refined inner region
        """
        # 1. Extract inner landmarks (main 17-67 -> inner 0-50)
        inner_landmarks = self._extract_inner_landmarks(main_landmarks)

        # 2. Fit inner PDM to get initial parameters
        inner_params = self.pdm.calc_params(inner_landmarks)

        # 3. Check scale requirement
        # C++: if (params_global[0] > 0.9 * patch_scaling[0])
        main_scale = main_params[0] if main_params is not None else 1.0
        if main_scale < self.min_scale_ratio * self.patch_scaling:
            # Face too small - skip refinement, return original
            return main_landmarks

        # 4. Run NU-RLMS optimization
        optimized_params, info = self.optimizer.optimize(
            self.pdm,
            inner_params,
            self.ccnf_experts,
            image
        )

        # 5. Get optimized landmarks and map back to main
        refined_inner = self.pdm.params_to_landmarks_2d(optimized_params)
        refined_main = self._map_to_main(main_landmarks.copy(), refined_inner)

        return refined_main

    def _extract_inner_landmarks(self, main_landmarks: np.ndarray) -> np.ndarray:
        """
        Extract inner landmarks from main 68-point model.

        Args:
            main_landmarks: Full 68-point landmarks (68, 2)

        Returns:
            inner_landmarks: Inner 51-point landmarks (51, 2)
        """
        inner_landmarks = np.zeros((51, 2), dtype=main_landmarks.dtype)

        for main_idx, inner_idx in self.MAIN_TO_INNER.items():
            inner_landmarks[inner_idx] = main_landmarks[main_idx]

        return inner_landmarks

    def _map_to_main(self, main_landmarks: np.ndarray,
                     refined_inner: np.ndarray) -> np.ndarray:
        """
        Map refined inner landmarks back to main model.

        Args:
            main_landmarks: Full 68-point landmarks (to be modified)
            refined_inner: Refined 51-point inner landmarks

        Returns:
            main_landmarks: Updated 68-point landmarks
        """
        for inner_idx, main_idx in self.INNER_TO_MAIN.items():
            main_landmarks[main_idx] = refined_inner[inner_idx]

        return main_landmarks

    def get_inner_landmark_indices(self):
        """Get the main model indices that are refined by the inner model."""
        return list(range(17, 68))


def test_hierarchical_inner_model():
    """Test hierarchical inner model."""
    import cv2

    print("=" * 60)
    print("Testing Hierarchical Inner Model")
    print("=" * 60)

    model_dir = "pyclnf/pyclnf/models"
    inner_model = HierarchicalInnerModel(model_dir)

    print(f"\nInner model loaded:")
    print(f"  PDM points: {inner_model.pdm.n_points}")
    print(f"  PDM modes: {inner_model.pdm.n_modes}")
    print(f"  CCNF experts: {len(inner_model.ccnf_experts.experts)}")
    print(f"  Optimizer reg_factor: {inner_model.optimizer.reg_factor}")
    print(f"  Optimizer sigma: {inner_model.optimizer.sigma}")

    # Create synthetic test data
    test_image = np.random.randint(100, 200, (500, 500), dtype=np.uint8)

    # Create fake main landmarks (positioned around center)
    main_landmarks = np.zeros((68, 2))
    for i in range(68):
        angle = 2 * np.pi * i / 68
        radius = 100 + 30 * np.sin(3 * angle)
        main_landmarks[i] = [250 + radius * np.cos(angle),
                             250 + radius * np.sin(angle)]

    # Create fake main params
    main_params = np.zeros(40)
    main_params[0] = 2.0  # scale

    print(f"\nTest input:")
    print(f"  Image shape: {test_image.shape}")
    print(f"  Main landmarks shape: {main_landmarks.shape}")
    print(f"  Main scale: {main_params[0]}")

    # Test extraction
    inner_landmarks = inner_model._extract_inner_landmarks(main_landmarks)
    print(f"\nExtracted inner landmarks shape: {inner_landmarks.shape}")

    # Test refinement
    refined_landmarks = inner_model.refine_landmarks(test_image, main_landmarks, main_params)
    print(f"\nRefined landmarks shape: {refined_landmarks.shape}")

    # Check which landmarks changed
    diff = np.linalg.norm(refined_landmarks - main_landmarks, axis=1)
    changed_indices = np.where(diff > 0.01)[0]
    unchanged_indices = np.where(diff <= 0.01)[0]

    print(f"\nLandmark changes:")
    print(f"  Changed landmarks: {len(changed_indices)}")
    print(f"  Unchanged landmarks: {len(unchanged_indices)}")
    print(f"  Changed indices: {list(changed_indices[:10])}...")
    print(f"  Expected changed: 17-67 (inner face)")

    # Verify only inner landmarks changed
    inner_indices = set(range(17, 68))
    changed_set = set(changed_indices)
    if changed_set.issubset(inner_indices):
        print("  Correct: Only inner landmarks (17-67) were modified")
    else:
        outer_changed = changed_set - inner_indices
        print(f"  Warning: Outer landmarks were modified: {outer_changed}")

    print("\n" + "=" * 60)
    print("Hierarchical Inner Model tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_hierarchical_inner_model()
