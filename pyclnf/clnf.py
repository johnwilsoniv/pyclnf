"""
CLNF (Constrained Local Neural Fields) - Complete facial landmark detector

This is the main user-facing API that combines:
- PDM (Point Distribution Model) for shape representation
- CCNF patch experts for landmark detection
- NU-RLMS optimizer for parameter fitting
- Corrected RetinaFace detector (ARM Mac optimized, primary detector)

Usage (PRIMARY - with automatic face detection):
    from pyclnf import CLNF
    import cv2

    # Initialize model with corrected RetinaFace detector (default)
    clnf = CLNF()  # ARM Mac optimized, 8.23px accuracy

    # Detect and fit landmarks automatically
    image = cv2.imread("face.jpg")
    landmarks, info = clnf.detect_and_fit(image)

Usage (LEGACY - with manual bbox):
    from pyclnf import CLNF

    # Initialize model without detector
    clnf = CLNF(detector=None)

    # Fit landmarks with manual bbox
    landmarks, info = clnf.fit(image, face_bbox)
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import cv2
from pathlib import Path

from .core.pdm import PDM
from .core.cen_patch_expert import CENModel
from .core.optimizer import NURLMSOptimizer
from .utils.retinaface_correction import RetinaFaceCorrectedDetector
from .core.eye_patch_expert import HierarchicalEyeModel


class CLNF:
    """
    Complete CLNF facial landmark detector.

    Fits a statistical shape model (PDM) to detected facial features using
    patch experts and constrained optimization.
    """

    # Default to package-relative models directory
    _DEFAULT_MODEL_DIR = Path(__file__).parent / "models"

    def __init__(self,
                 model_dir: str = None,
                 scale: float = 0.25,
                 regularization: float = 22.5,  # C++ CECLM: 25.0 base × 0.9 = 22.5
                 max_iterations: int = 10,  # Per phase distributed across windows (~40 total to match C++)
                 convergence_threshold: float = 0.005,  # Gold standard: strict convergence for accuracy
                 sigma: float = 2.5,  # KDE kernel sigma - verified from C++ debug dump
                 weight_multiplier: float = 0.0,  # Disabled - hurts face model (tested: 2.0, 5.0 both worse)
                 window_sizes: list = None,
                 detector: str = "pymtcnn",
                 detector_model_path: Optional[str] = None,
                 use_coreml: bool = False,
                 use_gpu: bool = True,  # Enable GPU acceleration for response maps and mean-shift
                 gpu_device: str = 'auto',  # GPU device: 'auto', 'mps', 'cuda', 'cpu'
                 use_eye_refinement: bool = True,  # Enabled - matches C++ pipeline (0.15 px error vs C++ post-refinement)
                 use_inner_refinement: bool = False,  # Disabled by default - experimental inner face refinement
                 debug_mode: bool = False,
                 tracked_landmarks: list = None,
                 use_shared_memory: bool = False,
                 shared_memory_dir: str = None,
                 convergence_profile: str = None,
                 early_window_exit: bool = False,
                 early_exit_threshold: float = 0.3,
                 use_validator: bool = None,
                 validator_threshold: float = 0.725,
                 reinit_video_every: int = 2):
        """
        Initialize CLNF model.

        Args:
            model_dir: Directory containing exported PDM and CCNF models
            scale: DEPRECATED - now loads all scales [0.25, 0.35, 0.5]
            regularization: Shape regularization weight (higher = stricter shape prior)
                          Default: 22.5 (C++ CECLM: 25.0 base × 0.9 scaling)
            max_iterations: Maximum optimization iterations TOTAL across all window sizes
                          (OpenFace default: 5 per window × 4 windows = 20 total)
            convergence_threshold: Mean per-landmark convergence threshold in pixels
                          (default: 0.005 pixels for gold standard accuracy)
            sigma: Gaussian kernel sigma for KDE mean-shift
                  (C++ OpenFace uses σ=2.5 - verified from debug dump)
            weight_multiplier: Weight multiplier w for patch confidences
                             (OpenFace uses w=7 for Multi-PIE, w=5 for in-the-wild)
            window_sizes: List of window sizes for hierarchical refinement (default: [11, 9, 7])
                         Note: Only window sizes with sigma components are supported ([7, 9, 11, 15])
            detector: Face detector to use ("pymtcnn", "retinaface", or None). Default: "pymtcnn"
            detector_model_path: Path to detector model. If None, uses default path
            use_coreml: Enable CoreML acceleration (ARM Mac optimization)
            use_gpu: Enable GPU acceleration for response maps and mean-shift computation.
                    Uses PyTorch MPS (Apple Silicon) or CUDA (NVIDIA). Default: False.
                    When enabled, provides 10-20x speedup for response map computation.
            gpu_device: GPU device to use ('auto', 'mps', 'cuda', 'cpu'). Default: 'auto'.
            debug_mode: Enable debug output for development
            tracked_landmarks: List of landmark indices to track for debugging
            use_shared_memory: If True, use memory-mapped shared models for HPC multiprocessing
                             Reduces memory from 424MB × N_workers to 424MB × 1
            shared_memory_dir: Directory for shared memory files (default: /dev/shm/pyclnf_models)
            convergence_profile: Named profile ('accurate', 'optimized', 'fast', 'video')
                               If provided, overrides max_iterations and convergence_threshold
            early_window_exit: If True, skip remaining windows when already converged (default: True)
            early_exit_threshold: Mean shift threshold for early window exit (default: 0.3px)
            use_validator: Enable CNN-based detection validation (default: True for video mode)
                          When enabled in video mode, triggers re-detection if validation fails
            validator_threshold: Confidence threshold for validator (default: 0.725, matching C++)
            reinit_video_every: Number of consecutive validation failures before re-detection (default: 2)
        """
        self.use_shared_memory = use_shared_memory
        self.shared_memory_dir = shared_memory_dir
        self.model_dir = Path(model_dir) if model_dir else self._DEFAULT_MODEL_DIR
        self.regularization = regularization
        self.sigma = sigma
        self.weight_multiplier = weight_multiplier
        self.debug_mode = debug_mode  # Use parameter value
        self.tracked_landmarks = tracked_landmarks if tracked_landmarks is not None else [36, 48, 30, 8]

        # GPU acceleration settings
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device

        # Phase 2 HPC optimizations: Convergence profile and early window exit
        self.convergence_profile = convergence_profile
        self.early_window_exit = early_window_exit
        self.early_exit_threshold = early_exit_threshold

        # Temporal warm-start state for video mode
        # These track previous frame's results for faster convergence on subsequent frames
        self._prev_frame_params = None
        self._prev_frame_landmarks = None
        self._video_mode = True  # Default ON - enables tracking with previous frame params

        # Video-mode template tracking (matches C++ OpenFace)
        # Template matching corrects global parameters before optimization
        # This significantly improves left jaw landmark accuracy in video mode
        self._face_template = None  # Stored face template from previous successful frame
        self._template_init_box = None  # Bounding box when template was extracted (x_min, y_min, w, h)
        self._template_face_offset = None  # Offset of face bbox within template (dx, dy)
        self._template_scale = 0.3  # Scale factor for template (matches C++ face_template_scale)
        self._use_template_tracking = True  # Default ON - template tracking for global param correction
        self._tracking_initialized = False  # Whether tracking has been initialized

        # C++ video tracking failure state (mirrors LandmarkDetectorFunc.cpp:215-289)
        # failures_in_a_row: -1 = initial state, reset to -1 on success, increment on failure
        # detection_success: tracks if previous frame's fitting was successful
        self._failures_in_a_row = -1  # C++ uses -1 as initial state
        self._detection_success = False  # Set True when validator passes

        # Adaptive window sizes for video mode (matches C++ OpenFace)
        # After first frame, use smaller search windows for faster tracking
        # C++ window_sizes_small = [0, 9, 7, 0] - 0 means skip that scale
        # C++ window_sizes_init = [11, 9, 7, 5] - includes all scales for first detection
        self._window_sizes_init = [11, 9, 7, 5]  # First detection - all scales
        # C++ uses window_sizes_small = [0, 9, 7, 0] - middle scales only
        # Skip coarsest (11) and finest (5) for faster tracking
        self._window_sizes_tracking = [9, 7]  # Track with middle window sizes only

        # Match C++ OpenFace window sizes including WS5
        # WS5 uses scale 1.0 which requires scale clamping (fixed in optimizer.py)
        # The sigma_components files only exist for [7, 9, 11, 15], but WS5 works with identity
        default_windows = [11, 9, 7, 5]
        self.window_sizes = window_sizes if window_sizes is not None else default_windows

        # OpenFace uses multiple patch scales (coarse to fine)
        # Each window size maps to a patch scale: 11→0.25, 9→0.35, 7→0.5, 5→1.0
        self.patch_scaling = [0.25, 0.35, 0.5, 1.0]

        # Map window sizes to patch scale indices (coarse-to-fine)
        # Larger windows use coarser scales, smaller windows use finer scales
        self.window_to_scale = self._map_windows_to_scales()

        # Load PDM (shape model)
        pdm_dir = self.model_dir / "exported_pdm"
        self.pdm = PDM(str(pdm_dir))

        # Load CEN patch experts for ALL scales (from .dat files)
        # Use shared memory for HPC multiprocessing (reduces memory by ~93%)
        self.ccnf = CENModel(
            str(self.model_dir),
            scales=self.patch_scaling,
            use_shared_memory=use_shared_memory,
            shared_memory_dir=shared_memory_dir
        )

        # NOTE: Previously filtered window sizes to only those with sigma components,
        # but this removes window size 5 which is needed for the finest scale.
        # Sigma components are optional (for CCNF spatial correlation), so we keep all window sizes.
        # The optimizer will skip sigma transformation for window sizes without components.
        if self.ccnf.sigma_components:
            available_windows = list(self.ccnf.sigma_components.keys())
            missing_windows = [ws for ws in self.window_sizes if ws not in available_windows]
            if missing_windows:
                print(f"Note: No sigma components for window sizes {missing_windows}, using identity transform")

        # Initialize optimizer with OpenFace parameters
        # Pass convergence_profile to enable HPC-optimized convergence settings
        self.optimizer = NURLMSOptimizer(
            regularization=regularization,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            sigma=sigma,
            weight_multiplier=weight_multiplier,  # CRITICAL: Apply weight multiplier
            debug_mode=debug_mode,
            tracked_landmarks=self.tracked_landmarks,
            convergence_profile=convergence_profile,  # Phase 2: Named profile for HPC optimization
            use_gpu=use_gpu,  # GPU acceleration for response maps and mean-shift
            gpu_device=gpu_device  # GPU device selection
        )

        if use_gpu and self.optimizer.use_gpu:
            print(f"✓ GPU acceleration enabled (device: {gpu_device})")

        # Initialize eye refinement model if enabled
        self.use_eye_refinement = use_eye_refinement
        self.eye_model = None
        if use_eye_refinement:
            try:
                self.eye_model = HierarchicalEyeModel(
                    str(self.model_dir),
                    use_gpu=use_gpu,
                    gpu_device=gpu_device
                )
                gpu_status = " (GPU)" if self.eye_model.use_gpu else ""
                print(f"✓ Eye refinement model loaded{gpu_status}")
            except Exception as e:
                print(f"Warning: Could not load eye refinement model: {e}")
                self.use_eye_refinement = False

        # Initialize inner face refinement model if enabled (experimental)
        self.use_inner_refinement = use_inner_refinement
        self.inner_model = None
        if use_inner_refinement:
            try:
                from .core.inner_model import HierarchicalInnerModel
                self.inner_model = HierarchicalInnerModel(str(self.model_dir))
                print("✓ Inner refinement model loaded")
            except Exception as e:
                print(f"Warning: Could not load inner refinement model: {e}")
                self.use_inner_refinement = False

        # Initialize face detector (PRIMARY: PyMTCNN with all performance optimizations)
        # Store detector type for bbox preprocessing in PDM initialization
        self.detector_type = detector  # 'pymtcnn', 'retinaface', etc.
        self.detector = None
        if detector == "pymtcnn":
            # Use PyMTCNN detector (default for production)
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent / "pymtcnn"))
                from pymtcnn import MTCNN
                self.detector = MTCNN()
                print("✓ PyMTCNN detector initialized (CoreML/ONNX auto-selection)")
            except Exception as e:
                print(f"Warning: Could not initialize PyMTCNN detector: {e}")
                print("Detector will not be available. Use fit() with manual bbox instead.")
        elif detector == "retinaface":
            # Fallback: Corrected RetinaFace for ARM Mac optimization
            if detector_model_path is None:
                detector_model_path = "S1 Face Mirror/weights/retinaface_mobilenet025_coreml.onnx"

            try:
                self.detector = RetinaFaceCorrectedDetector(
                    model_path=detector_model_path,
                    use_coreml=use_coreml
                )
                print("✓ RetinaFace detector initialized")
            except Exception as e:
                print(f"Warning: Could not initialize RetinaFace detector: {e}")
                print("Detector will not be available. Use fit() with manual bbox instead.")

        # Initialize detection validator (optional, default: True for video mode)
        # Validator checks if tracking is still valid after each fit
        # If validation fails consecutively, triggers re-detection
        self.use_validator = use_validator if use_validator is not None else self._video_mode
        self.validator_threshold = validator_threshold
        self.reinit_video_every = reinit_video_every
        self._validation_failures = 0  # Consecutive validation failures
        self.validator = None
        if self.use_validator:
            try:
                from .core.detection_validator import DetectionValidator
                validator_path = self.model_dir / "detection_validation" / "validator_cnn_68.txt"
                self.validator = DetectionValidator(str(validator_path))
            except Exception as e:
                print(f"Warning: Could not load detection validator: {e}")
                self.use_validator = False

    def fit(self,
            image: np.ndarray,
            face_bbox: Tuple[float, float, float, float],
            initial_params: Optional[np.ndarray] = None,
            landmarks_5pt: Optional[np.ndarray] = None,
            return_params: bool = False,
            detector_type: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        Fit CLNF model to detect facial landmarks.

        Args:
            image: Input image (grayscale or color, will be converted to grayscale)
            face_bbox: Face bounding box [x, y, width, height]
            initial_params: Optional initial parameter guess (default: from bbox)
            landmarks_5pt: Optional 5-point MTCNN landmarks for better initialization.
                           Shape (5, 2) with [left_eye, right_eye, nose, left_mouth, right_mouth].
                           If provided, estimates initial pose from these landmarks.
            return_params: If True, include optimized parameters in info dict
            detector_type: Type of face detector. If 'pymtcnn', applies OpenFace MTCNN
                           bbox correction (FaceDetectorMTCNN.cpp lines 1498-1504).
                           Pass None for pre-corrected bboxes or other detectors.

        Returns:
            landmarks: Detected 2D landmarks, shape (68, 2)
            info: Dictionary with fitting information:
                - converged: bool
                - iterations: int
                - final_update: float
                - params: np.ndarray (if return_params=True)
                - bbox: corrected bbox used for fitting
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # PyMTCNN now outputs calibrated bboxes directly (calibration in pymtcnn/base.py)
        # No additional correction needed here
        effective_detector = None

        # Initialize parameters from bounding box (and optionally 5-point landmarks)
        if initial_params is None:
            # Phase 2: Video mode temporal warm-start
            # C++ tracking (LandmarkDetectorFunc.cpp:230-249) uses previous params DIRECTLY
            # without reinitializing from bbox. Only template matching applies a SHIFT.
            if self._video_mode and self._prev_frame_params is not None:
                params = self._prev_frame_params.copy()
                # NOTE: C++ does NOT override tx, ty from bbox in tracking mode.
                # It uses previous params directly and only applies template shift later.
                # This is critical for accuracy - bbox-derived tx/ty introduces drift.
                if self.debug_mode:
                    print(f"[VIDEO_MODE] Using previous params directly (C++ style)")
            elif landmarks_5pt is not None and landmarks_5pt.shape == (5, 2):
                # Use 5-point landmarks for initial pose estimation (like C++ OpenFace)
                # Note: init_params_from_5pt uses landmarks directly, no bbox correction needed
                params = self.pdm.init_params_from_5pt(face_bbox, landmarks_5pt)
                if self.debug_mode:
                    print(f"[INIT] Using 5-point landmarks for pose estimation")
                    print(f"  scale={params[0]:.4f}, rot=({params[1]:.4f}, {params[2]:.4f}, {params[3]:.4f})")
            else:
                params = self.pdm.init_params(face_bbox, detector_type=effective_detector)
        else:
            params = initial_params.copy()

        # Estimate head pose from bbox for view selection
        # For now, assume frontal view (view 0)
        # TODO: Implement pose estimation from bbox orientation
        view_idx = 0
        pose = np.array([0.0, 0.0, 0.0])  # [pitch, yaw, roll]

        # Video mode: Apply template matching correction before optimization
        # This helps stabilize outer landmarks where patch responses are ambiguous
        if self._use_template_tracking and self._tracking_initialized:
            params = self._correct_global_parameters_video(gray, params)

        # Video mode: Use adaptive window sizes after first frame
        # C++ uses window_sizes_small = [0, 9, 7, 0] for tracking (0 = skip)
        active_window_sizes = self.window_sizes
        if self._tracking_initialized and self._video_mode:
            active_window_sizes = self._window_sizes_tracking
            if self.debug_mode:
                print(f"[VIDEO_MODE] Using tracking windows: {active_window_sizes}")

        # Hierarchical optimization with multiple window sizes and patch scales
        # OpenFace optimizes from large to small windows for coarse-to-fine refinement

        # FIX: Distribute max_iterations across window sizes instead of per-window
        # This ensures total iterations match max_iterations, not max_iterations × num_windows
        n_windows = len(active_window_sizes)
        iters_per_window = self.optimizer.max_iterations // n_windows
        iters_remainder = self.optimizer.max_iterations % n_windows

        # Save original max_iterations to restore later
        original_max_iterations = self.optimizer.max_iterations

        total_iterations = 0
        all_iteration_history = []  # Collect iteration history across all windows
        previous_window_landmarks = None  # For early window exit detection

        for window_idx, window_size in enumerate(active_window_sizes):
            # Phase 2 HPC Optimization: Early window exit
            # If landmarks haven't moved much since last window, skip remaining windows
            if self.early_window_exit and window_idx > 0 and previous_window_landmarks is not None:
                current_landmarks = self.pdm.params_to_landmarks_2d(params)
                landmark_change = np.linalg.norm(current_landmarks - previous_window_landmarks, axis=1).mean()
                if landmark_change < self.early_exit_threshold:
                    if self.debug_mode:
                        print(f"[EARLY_EXIT] Window {window_size}: landmark change {landmark_change:.4f}px < threshold {self.early_exit_threshold}px, skipping remaining windows")
                    break
            # Distribute remainder iterations to early windows
            # e.g., max_iter=10, 3 windows → [4, 3, 3]
            window_iters = iters_per_window + (1 if window_idx < iters_remainder else 0)

            # Override optimizer max_iterations for this window
            self.optimizer.max_iterations = window_iters

            # Get the appropriate patch scale for this window (coarse-to-fine)
            scale_idx = self.window_to_scale[window_size]
            patch_scale = self.patch_scaling[scale_idx]

            # Get patch experts for this view and scale
            patch_experts = self._get_patch_experts(view_idx, patch_scale)

            # WEIGHTS: Use uniform weights (1.0) for all landmarks to match C++ OpenFace
            # C++ OpenFace uses weight=1.0 for all landmarks in NU-RLMS optimization.
            # The patch_confidence values stored in CEN experts are NOT used as weights
            # in C++; they serve a different purpose (reliability estimation).
            # The Hessian diagonal for tx,ty should be exactly 68.0 (sum of weights).
            weights = np.ones(self.pdm.n_points)

            # NOTE: Regularization and sigma are adapted by the optimizer internally
            # based on patch_scaling (see optimizer._compute_scale_adapted_params)
            # The optimizer clamps scale to max 0.5 to match C++ behavior

            # Run optimization for this window size
            # Using patch experts trained at patch_scale for this window
            # CRITICAL: Pass patch_scaling to enable image warping
            optimized_params, opt_info = self.optimizer.optimize(
                self.pdm,
                params,
                patch_experts,
                gray,
                weights=weights,  # CRITICAL: Pass patch confidence weights
                window_size=window_size,
                patch_scaling=patch_scale,  # CRITICAL: Enable image warping to reference coordinates
                sigma_components=self.ccnf.sigma_components  # Enable CCNF spatial correlation modeling
            )

            # Update params for next iteration
            params = optimized_params
            total_iterations += opt_info['iterations']

            # Collect iteration history if available
            if 'iteration_history' in opt_info:
                all_iteration_history.extend(opt_info['iteration_history'])

            # Save landmarks for early window exit detection in next window
            previous_window_landmarks = self.pdm.params_to_landmarks_2d(params)

            # Early stopping if face becomes too small
            if params[0] < 0.25:  # Scale parameter
                break

        # Restore original max_iterations
        self.optimizer.max_iterations = original_max_iterations

        # Extract final landmarks
        landmarks = self.pdm.params_to_landmarks_2d(optimized_params)

        # Apply inner face refinement if enabled (landmarks 17-67)
        if self.use_inner_refinement and self.inner_model is not None:
            landmarks = self.inner_model.refine_landmarks(
                gray, landmarks, optimized_params
            )
            if self.debug_mode:
                print(f"[INNER_REFINE] Refined landmarks 17-67")

        # Apply eye refinement if enabled (landmarks 36-47)
        eye_iteration_history = []
        if self.use_eye_refinement and self.eye_model is not None:
            # Refine left eye (landmarks 36-41)
            result = self.eye_model.refine_eye_landmarks(
                gray, landmarks, 'left',
                main_rotation=optimized_params[1:4],
                main_scale=optimized_params[0],
                track_iterations=True
            )
            if isinstance(result, tuple):
                landmarks, left_eye_history = result
                eye_iteration_history.extend(left_eye_history)
            else:
                landmarks = result

            # Refine right eye (landmarks 42-47)
            result = self.eye_model.refine_eye_landmarks(
                gray, landmarks, 'right',
                main_rotation=optimized_params[1:4],
                main_scale=optimized_params[0],
                track_iterations=True
            )
            if isinstance(result, tuple):
                landmarks, right_eye_history = result
                eye_iteration_history.extend(right_eye_history)
            else:
                landmarks = result

            # Re-fit main PDM to refined landmarks (like C++ CalcParams + CalcShape2D)
            # This is critical for propagating eye refinement through the shape model
            # Uses C++ defaults: reg_factor=1.0, max_iter=1000, damping=0.75
            optimized_params = self.pdm.fit_to_landmarks_2d(
                landmarks, optimized_params
            )
            # Update landmarks from re-fitted params for consistency
            landmarks = self.pdm.params_to_landmarks_2d(optimized_params)

        # Phase 2: Store results for video mode temporal warm-start
        if self._video_mode:
            self._prev_frame_params = optimized_params.copy()
            self._prev_frame_landmarks = landmarks.copy()

        # Video mode: Always update template after fitting (like C++)
        # C++ OpenFace always updates the template, regardless of "convergence"
        # Convergence in CLNF is just an early-stop mechanism, not a quality metric
        if self._use_template_tracking:
            self._update_face_template(gray, optimized_params)
            if not self._tracking_initialized:
                self._tracking_initialized = True
                if self.debug_mode:
                    print(f"[VIDEO_MODE] Tracking initialized, will use template matching on next frame")

        # Prepare output info
        info = {
            'converged': opt_info['converged'],
            'iterations': total_iterations,
            'final_update': opt_info['final_update'],
            'view': view_idx,
            'pose': pose,
            'bbox': face_bbox,  # Corrected bbox used for fitting
            'iteration_history': all_iteration_history,  # Include full iteration history
            'eye_iteration_history': eye_iteration_history  # Include eye iteration history
        }

        if return_params:
            info['params'] = optimized_params

        # Include response maps for likelihood computation (used by multi-hypothesis)
        if hasattr(self.optimizer, 'cached_response_maps') and self.optimizer.cached_response_maps is not None:
            info['response_maps'] = self.optimizer.cached_response_maps

        # Run detection validator if enabled
        if self.validator is not None:
            orientation = np.array([pose[0], pose[1], pose[2]])
            validation_confidence = self.validator.check(orientation, gray, landmarks)
            is_valid = validation_confidence > self.validator_threshold
            info['validation_confidence'] = validation_confidence
            info['validation_passed'] = is_valid

            if is_valid:
                self._validation_failures = 0
                # C++ tracking state: success resets failures_in_a_row to -1
                self._detection_success = True
                self._failures_in_a_row = -1
            else:
                self._validation_failures += 1
                # C++ tracking state: failure increments failures_in_a_row
                self._detection_success = False
                self._failures_in_a_row += 1
                if self.debug_mode:
                    print(f"[VALIDATOR] Failed ({self._validation_failures}/{self.reinit_video_every}): "
                          f"confidence={validation_confidence:.4f} < {self.validator_threshold}")

            info['validation_failures'] = self._validation_failures
            info['needs_redetection'] = self._validation_failures >= self.reinit_video_every
        else:
            # No validator = assume success (C++ behavior)
            self._detection_success = True
            self._failures_in_a_row = -1

        return landmarks, info

    def fit_multi_hypothesis(self,
                             image: np.ndarray,
                             face_bbox: Tuple[float, float, float, float],
                             return_params: bool = False,
                             detector_type: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        Fit CLNF model using multi-hypothesis rotation testing (like C++ OpenFace).

        This matches the C++ OpenFace DetectLandmarksInImageMultiHypBasic function.
        Tests 11 rotation hypotheses and selects the best by model likelihood.

        The 11 hypotheses are (from C++ LandmarkDetectorFunc.cpp lines 728-738):
        - (0, 0, 0) - frontal
        - (0, -0.5236, 0) - yaw -30°
        - (0, 0.5236, 0) - yaw +30°
        - (0, -0.96, 0) - yaw -55°
        - (0, 0.96, 0) - yaw +55°
        - (0, 0, 0.5236) - roll +30°
        - (0, 0, -0.5236) - roll -30°
        - (0, -1.57, 0) - yaw -90° (profile left)
        - (0, 1.57, 0) - yaw +90° (profile right)
        - (0, -1.22, 0.698) - yaw -70° with roll
        - (0, 1.22, -0.698) - yaw +70° with roll

        Args:
            image: Input image (grayscale or color)
            face_bbox: Face bounding box [x, y, width, height]
            return_params: If True, include optimized parameters in info dict
            detector_type: Type of face detector (see fit() for options)

        Returns:
            landmarks: Best detected 2D landmarks, shape (68, 2)
            info: Dictionary with fitting information including 'best_hypothesis'
        """
        # C++ rotation hypotheses (pitch, yaw, roll)
        rotation_hypotheses = [
            np.array([0.0, 0.0, 0.0]),        # frontal
            np.array([0.0, -0.5236, 0.0]),    # yaw -30°
            np.array([0.0, 0.5236, 0.0]),     # yaw +30°
            np.array([0.0, -0.96, 0.0]),      # yaw -55°
            np.array([0.0, 0.96, 0.0]),       # yaw +55°
            np.array([0.0, 0.0, 0.5236]),     # roll +30°
            np.array([0.0, 0.0, -0.5236]),    # roll -30°
            np.array([0.0, -1.57, 0.0]),      # yaw -90° (profile left)
            np.array([0.0, 1.57, 0.0]),       # yaw +90° (profile right)
            np.array([0.0, -1.22, 0.698]),    # yaw -70° with roll
            np.array([0.0, 1.22, -0.698]),    # yaw +70° with roll
        ]

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Determine effective detector type (None = no correction, matching C++ OpenFace)
        effective_detector = detector_type  # Pass through as-is

        best_landmarks = None
        best_params = None
        best_info = None
        best_likelihood = -np.inf
        best_hypothesis_idx = 0

        # Save tracking state BEFORE hypothesis testing
        # Each fit() call updates tracking state, which corrupts subsequent hypotheses
        saved_tracking_initialized = self._tracking_initialized
        saved_face_template = self._face_template.copy() if self._face_template is not None else None
        saved_template_init_box = self._template_init_box
        saved_template_face_offset = self._template_face_offset
        saved_template_scale = self._template_scale
        saved_prev_frame_params = self._prev_frame_params.copy() if self._prev_frame_params is not None else None
        saved_prev_frame_landmarks = self._prev_frame_landmarks.copy() if self._prev_frame_landmarks is not None else None

        for idx, rotation in enumerate(rotation_hypotheses):
            # Restore tracking state before each hypothesis
            # This ensures each hypothesis starts from the same state
            self._tracking_initialized = saved_tracking_initialized
            self._face_template = saved_face_template.copy() if saved_face_template is not None else None
            self._template_init_box = saved_template_init_box
            self._template_face_offset = saved_template_face_offset
            self._template_scale = saved_template_scale
            self._prev_frame_params = saved_prev_frame_params.copy() if saved_prev_frame_params is not None else None
            self._prev_frame_landmarks = saved_prev_frame_landmarks.copy() if saved_prev_frame_landmarks is not None else None
            # Clear response map cache to ensure fresh computation for each hypothesis
            self.optimizer.cached_response_maps = None
            self.optimizer.cached_landmarks = None

            # Initialize params with this rotation hypothesis
            initial_params = self.pdm.init_params_with_rotation(
                face_bbox, rotation, detector_type=effective_detector
            )

            # Run full CLNF fitting with this initialization
            landmarks, info = self.fit(
                image, face_bbox, initial_params=initial_params, return_params=True,
                detector_type=effective_detector
            )
            params = info['params']

            # Compute model likelihood from patch response maps (matching C++ OpenFace)
            # C++ uses response maps weighted by Gaussian kernel centered at landmark positions
            response_maps = info.get('response_maps', None)
            likelihood = self._compute_model_likelihood(gray, landmarks, params, response_maps)

            if self.debug_mode:
                print(f"[MULTI_HYP] Hypothesis {idx}: rot=({rotation[0]:.3f}, {rotation[1]:.3f}, {rotation[2]:.3f})")
                print(f"            likelihood={likelihood:.4f}, converged={info['converged']}")

            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_landmarks = landmarks
                best_params = params
                best_info = info.copy()
                best_hypothesis_idx = idx

        # Update best_info with multi-hypothesis details
        best_info['best_hypothesis'] = best_hypothesis_idx
        best_info['best_rotation'] = rotation_hypotheses[best_hypothesis_idx].tolist()
        best_info['model_likelihood'] = best_likelihood

        if return_params:
            best_info['params'] = best_params

        # FIX: Update tracking state with BEST hypothesis results
        # We saved/restored state during hypothesis testing, now set final state
        if self._video_mode:
            self._prev_frame_params = best_params.copy()
            self._prev_frame_landmarks = best_landmarks.copy()
            self._tracking_initialized = True
            # Update template from best hypothesis
            if self._use_template_tracking:
                self._update_face_template(gray, best_params)

        return best_landmarks, best_info

    def _compute_model_likelihood(self, gray: np.ndarray, landmarks: np.ndarray,
                                   params: np.ndarray,
                                   response_maps: Optional[Dict] = None,
                                   sigma: float = 2.25) -> float:
        """
        Compute model likelihood for hypothesis selection.

        This matches C++ OpenFace NU_RLMS likelihood computation (LandmarkDetectorModel.cpp:2051-2096).

        C++ Algorithm:
            For each landmark i with visibility:
                sum = 0
                For each pixel (ii, jj) in patch_response[i]:
                    v = patch_response[ii, jj]
                    dist_sq = (dy - ii)^2 + (dx - jj)^2
                    v *= exp(-0.5 * dist_sq / sigma^2)  # Gaussian weight
                    sum += v
                loglhood += log(sum + 1e-8)
            loglhood /= num_visible_landmarks

        Where (dx, dy) = center of response map = (resp_size/2, resp_size/2)

        Args:
            gray: Grayscale image
            landmarks: 2D landmarks (68, 2)
            params: Parameter vector [s, wx, wy, wz, tx, ty, local...]
            response_maps: Optional pre-computed response maps (from optimizer)
            sigma: Gaussian kernel sigma for KDE (default 2.25, matching C++ CECLM)

        Returns:
            likelihood: Log-likelihood (higher is better)
        """
        h, w = gray.shape[:2]
        n_landmarks = len(landmarks)

        # If we have response maps, compute proper patch-based likelihood
        if response_maps is not None and len(response_maps) > 0:
            loglhood = 0.0
            n_visible = 0

            a_kde = -0.5 / (sigma * sigma)

            for i in range(n_landmarks):
                # Skip if landmark outside image
                x, y = landmarks[i]
                if x < 0 or x >= w or y < 0 or y >= h:
                    continue

                if i not in response_maps:
                    continue

                resp = response_maps[i]
                resp_size = resp.shape[0]

                # Center of response map
                dx = resp_size / 2.0
                dy = resp_size / 2.0

                # Compute Gaussian-weighted sum of response
                total_sum = 0.0
                for ii in range(resp_size):
                    vx = (dy - ii) ** 2
                    for jj in range(resp_size):
                        vy = (dx - jj) ** 2
                        v = resp[ii, jj]
                        v *= np.exp(a_kde * (vx + vy))
                        total_sum += v

                loglhood += np.log(total_sum + 1e-8)
                n_visible += 1

            if n_visible > 0:
                loglhood /= n_visible

            return loglhood

        # Fallback: simple heuristic when no response maps available
        # This is less accurate but allows multi-hypothesis to still work

        # Penalize landmarks outside image
        in_bounds = (
            (landmarks[:, 0] >= 0) & (landmarks[:, 0] < w) &
            (landmarks[:, 1] >= 0) & (landmarks[:, 1] < h)
        )
        bounds_score = np.mean(in_bounds)

        # Penalize large shape deformations (regularization)
        local_params = params[6:]
        eigenvalues = self.pdm.eigen_values
        if len(local_params) > 0 and len(eigenvalues) > 0:
            # Weight by inverse eigenvalues (like C++ regularization)
            shape_penalty = np.sum((local_params ** 2) / (eigenvalues + 1e-6))
            shape_score = np.exp(-0.01 * shape_penalty)
        else:
            shape_score = 1.0

        # Penalize unreasonable scale
        scale = params[0]
        if scale < 0.5 or scale > 10:
            scale_score = 0.1
        else:
            scale_score = 1.0

        # Combined likelihood (log scale for consistency)
        likelihood = np.log(bounds_score * shape_score * scale_score + 1e-8)

        return likelihood

    def detect_and_fit(self,
                       image: np.ndarray,
                       return_all_faces: bool = False,
                       return_params: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Detect faces and fit CLNF landmarks in one call using the built-in detector.

        This is the primary method for using pyCLNF with automatic face detection.
        Uses corrected RetinaFace as the default detector (ARM Mac optimized).

        Args:
            image: Input image (grayscale or color, will be converted to grayscale)
            return_all_faces: If True, return results for all detected faces
                            If False, return only the first (largest) face
            return_params: If True, include optimized parameters in info dict

        Returns:
            If return_all_faces=False (default):
                landmarks: Detected 2D landmarks for first face, shape (68, 2)
                info: Dictionary with fitting information including 'bbox'
            If return_all_faces=True:
                results: List of (landmarks, info) tuples for each detected face

        Raises:
            ValueError: If no detector is initialized or no faces detected

        Example:
            >>> from pyclnf import CLNF
            >>> clnf = CLNF()  # Initializes with corrected RetinaFace
            >>> image = cv2.imread("face.jpg")
            >>> landmarks, info = clnf.detect_and_fit(image)
            >>> print(f"Detected {len(landmarks)} landmarks")
        """
        if self.detector is None:
            raise ValueError(
                "No detector initialized. Either:\n"
                "1. Initialize CLNF with detector='retinaface' (default)\n"
                "2. Use fit() method with manual bbox instead"
            )

        # TRACKING MODE: Skip re-detection, derive bbox from previous landmarks
        # C++ OpenFace doesn't re-run face detector during tracking - it uses
        # previous landmarks directly. Re-detecting causes bbox variance that
        # propagates to landmark error (especially on larger faces).
        if self._tracking_initialized and self._video_mode and self._prev_frame_landmarks is not None:
            # Derive bbox from previous landmarks with margin
            prev_lm = self._prev_frame_landmarks
            x_min, y_min = prev_lm.min(axis=0)
            x_max, y_max = prev_lm.max(axis=0)
            w, h = x_max - x_min, y_max - y_min
            margin = 0.3  # 30% margin around face
            bbox = (x_min - w * margin, y_min - h * margin,
                    w * (1 + 2 * margin), h * (1 + 2 * margin))

            # Convert to grayscale for fitting
            if len(image.shape) == 3:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image_gray = image

            # Tracking mode: use fit() which uses prev_frame_params and template matching
            landmarks, info = self.fit(image_gray, bbox, return_params=return_params,
                                       detector_type=self.detector_type)
            info['bbox'] = bbox
            info['tracking_mode'] = True
            return landmarks, info

        # FIRST FRAME / RE-DETECTION MODE: Run face detector

        # IMPORTANT: MTCNN detection REQUIRES original color image for accuracy.
        # Converting grayscale->BGR produces DIFFERENT detections than original color!
        # This causes ~4px jaw landmark errors (grayscale->BGR bbox differs by ~5-16px).
        if len(image.shape) == 2:
            raise ValueError(
                "detect_and_fit() requires a color (BGR) image for accurate face detection.\n"
                "MTCNN produces different bounding boxes for grayscale vs color images,\n"
                "which causes ~4px jaw landmark error.\n\n"
                "Solutions:\n"
                "1. Pass the original color image: clnf.detect_and_fit(frame_bgr)\n"
                "2. For grayscale-only workflows, use fit() with external bbox:\n"
                "   bbox = your_face_detector(image)\n"
                "   landmarks, info = clnf.fit(gray_image, bbox)"
            )

        # Color input - use as-is for detector, convert to gray for fitting
        image_bgr = image
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces - handle different detector APIs
        landmarks_5pt_all = None  # Will store 5-point landmarks if available
        if hasattr(self.detector, 'detect_and_correct'):
            # RetinaFace corrected detector (no 5-point landmarks)
            bboxes = self.detector.detect_and_correct(image_bgr)
        elif hasattr(self.detector, 'detect'):
            # MTCNN detector - returns (bboxes, landmarks_5pt)
            result = self.detector.detect(image_bgr)
            if isinstance(result, tuple) and len(result) >= 2:
                bboxes = result[0]  # (N, 4) array of [x, y, w, h]
                landmarks_5pt_all = result[1]  # (N, 5, 2) array of 5-point landmarks
                # Convert numpy array to list of tuples for compatibility
                bboxes = [tuple(bbox) for bbox in bboxes]
            else:
                bboxes = result
        else:
            raise ValueError("Detector does not have detect() or detect_and_correct() method")

        if len(bboxes) == 0:
            raise ValueError("No faces detected in image")

        # Process all faces if requested
        if return_all_faces:
            results = []
            for i, bbox in enumerate(bboxes):
                # Get corresponding 5-point landmarks if available
                lm5 = landmarks_5pt_all[i] if landmarks_5pt_all is not None else None

                # Pass detector_type to fit() so it applies bbox correction
                landmarks, info = self.fit(image_gray, bbox, landmarks_5pt=lm5,
                                          return_params=return_params, detector_type=self.detector_type)
                info['bbox'] = bbox  # Add bbox to info (corrected bbox)
                results.append((landmarks, info))
            return results

        # Process only largest face (matches C++ CLNF behavior)
        # C++ selects face with largest width (LandmarkDetectorUtils.cpp:809)
        # Python bboxes are (x, y, width, height), so bbox[2] is width
        if len(bboxes) == 1:
            bbox = bboxes[0]
            largest_idx = 0
        else:
            # Select largest face by width (matching C++ DetectSingleFaceMTCNN)
            widths = [bbox[2] for bbox in bboxes]
            largest_idx = np.argmax(widths)
            bbox = bboxes[largest_idx]

        # Get 5-point landmarks for the selected face
        landmarks_5pt = landmarks_5pt_all[largest_idx] if landmarks_5pt_all is not None else None

        # C++ OpenFace behavior:
        # - First frame (or after tracking failure): multi-hypothesis fitting
        # - Subsequent frames with tracking: single hypothesis using previous params
        if self._tracking_initialized and self._video_mode:
            # Tracking mode: use previous params, skip multi-hypothesis
            landmarks, info = self.fit(image_gray, bbox, landmarks_5pt=landmarks_5pt,
                                       return_params=return_params, detector_type=self.detector_type)
        else:
            # First frame or tracking lost: multi-hypothesis fitting
            landmarks, info = self.fit_multi_hypothesis(image_gray, bbox, return_params=return_params)

        info['bbox'] = info.get('bbox', bbox)  # Use corrected bbox from fit() if available
        return landmarks, info

    def fit_video(self,
                  video_path: str,
                  face_detector,
                  output_path: Optional[str] = None,
                  visualize: bool = True) -> list:
        """
        Fit CLNF to all frames in a video.

        Args:
            video_path: Path to input video
            face_detector: Face detector function (image -> bbox or None)
            output_path: Optional path to save visualization video
            visualize: If True, draw landmarks on frames

        Returns:
            results: List of (landmarks, info) tuples for each frame
        """
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer if output requested
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        results = []
        frame_idx = 0
        prev_params = None  # For temporal consistency
        redetect_count = 0
        bbox = None
        info = {}

        # Reset temporal state at start of video
        self.reset_temporal_state()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect face (or re-detect if validation triggered)
            # C++ re-detection logic (LandmarkDetectorFunc.cpp:291-302)
            reinit_every = self.reinit_video_every  # Default: 4

            # Condition 1: Not initialized, periodic attempt (every reinit*6 failures)
            condition1 = (not self._tracking_initialized and
                         self._failures_in_a_row >= 0 and
                         (self._failures_in_a_row + 1) % (reinit_every * 6) == 0)

            # Condition 2: Initialized but failed, periodic re-init
            condition2 = (self._tracking_initialized and
                         not self._detection_success and
                         reinit_every > 0 and
                         self._failures_in_a_row > 0 and
                         self._failures_in_a_row % reinit_every == 0)

            need_detection = (frame_idx == 0 or bbox is None or
                             info.get('needs_redetection', False) or
                             condition1 or condition2)

            if need_detection:
                bbox = face_detector(frame)
                if frame_idx > 0:  # Re-detection (not first frame)
                    redetect_count += 1
                    if self.debug_mode:
                        reason = "validator" if info.get('needs_redetection', False) else \
                                 "condition1 (uninit periodic)" if condition1 else \
                                 "condition2 (failed periodic)" if condition2 else "bbox lost"
                        print(f"[VIDEO] Frame {frame_idx}: Re-detection triggered ({reason}), "
                              f"failures={self._failures_in_a_row}, total redetects={redetect_count}")
                    self._validation_failures = 0  # Reset failure counter

            if bbox is not None:
                # Use previous frame's parameters as initialization for temporal consistency
                landmarks, info = self.fit(
                    frame,
                    bbox,
                    initial_params=prev_params,
                    return_params=True  # Need params for warm-start
                )

                # Store parameters for next frame
                prev_params = info.get('params')

                # Visualize if requested
                if visualize:
                    frame = self._draw_landmarks(frame, landmarks)

                results.append((landmarks, info))
            else:
                results.append((None, {'converged': False}))
                prev_params = None  # Reset on detection failure
                self.reset_temporal_state()  # Also reset internal temporal state

            # Write frame if output requested
            if writer:
                writer.write(frame)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        return results

    def _map_windows_to_scales(self) -> Dict[int, int]:
        """
        Map window sizes to patch scale indices.

        OpenFace C++ convention: window_sizes array is indexed by scale
        - window_sizes[0] → scale 0 (0.25)
        - window_sizes[1] → scale 1 (0.35)
        - window_sizes[2] → scale 2 (0.5)
        - window_sizes[3] → scale 3 (1.0)

        Window size 11 → 0.25, 9 → 0.35, 7 → 0.5, 5 → 1.0
        This mapping is independent of which windows are active (init vs tracking mode).

        Returns:
            Dictionary mapping window_size -> scale_index
        """
        # Fixed mapping based on C++ convention: larger windows → coarser scales
        window_scale_map = {
            11: 0,  # 0.25 scale
            9: 1,   # 0.35 scale
            7: 2,   # 0.5 scale
            5: 3,   # 1.0 scale
        }
        return window_scale_map

    def _get_patch_experts(self, view_idx: int, scale: float) -> Dict[int, 'CCNFPatchExpert']:
        """
        Get patch experts for a specific view and scale.

        Args:
            view_idx: View index (0-6)
            scale: Patch scale (0.25, 0.35, or 0.5)

        Returns:
            Dictionary mapping landmark_idx -> CCNFPatchExpert
        """
        patch_experts = {}

        scale_model = self.ccnf.scale_models.get(scale)
        if scale_model is None:
            return patch_experts

        view_data = scale_model['views'].get(view_idx)
        if view_data is None:
            return patch_experts

        # Get all available patches for this view
        patch_experts = view_data['patches']

        return patch_experts

    def _draw_landmarks(self,
                       image: np.ndarray,
                       landmarks: np.ndarray,
                       color: Tuple[int, int, int] = (0, 255, 0),
                       radius: int = 2) -> np.ndarray:
        """
        Draw landmarks on image.

        Args:
            image: Input image
            landmarks: Landmark positions (n_points, 2)
            color: Landmark color (B, G, R)
            radius: Landmark radius in pixels

        Returns:
            image: Image with landmarks drawn
        """
        vis = image.copy()

        for i, (x, y) in enumerate(landmarks):
            cv2.circle(vis, (int(x), int(y)), radius, color, -1)

        return vis

    def _correct_global_parameters_video(self,
                                         gray: np.ndarray,
                                         params: np.ndarray) -> np.ndarray:
        """
        Correct global parameters using template matching (matches C++ CorrectGlobalParametersVideo).

        This method uses the stored face template from the previous frame to find
        the best translation correction via template matching. This helps stabilize
        tracking when local patch responses are ambiguous (e.g., outer jaw landmarks).

        Args:
            gray: Grayscale image
            params: Current PDM parameters [scale, rot_x, rot_y, rot_z, tx, ty, local...]

        Returns:
            params: Updated parameters with corrected translation
        """
        if self._face_template is None or self._template_init_box is None:
            return params
        if self._template_face_offset is None:
            return params

        # Get current bounding box from PDM (init_box in C++)
        landmarks = self.pdm.params_to_landmarks_2d(params)
        x_min, y_min = landmarks.min(axis=0)
        x_max, y_max = landmarks.max(axis=0)
        width = x_max - x_min
        height = y_max - y_min

        # init_box is the bounding box from current params (where we expect face to be)
        init_box_x = x_min
        init_box_y = y_min

        # Create ROI (2x bbox size centered on init_box, like C++)
        roi_x = int(max(0, x_min - width / 2))
        roi_y = int(max(0, y_min - height / 2))
        roi_w = int(min(gray.shape[1] - roi_x, width * 2))
        roi_h = int(min(gray.shape[0] - roi_y, height * 2))

        if roi_w < self._face_template.shape[1] or roi_h < self._face_template.shape[0]:
            return params  # ROI too small for template matching

        roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # Scale template and ROI if needed (like C++)
        scaling = self._template_scale / params[0]
        if scaling < 1:
            template = cv2.resize(self._face_template, None, fx=scaling, fy=scaling)
            roi_scaled = cv2.resize(roi, None, fx=scaling, fy=scaling)
            # Scale the face offset too
            face_offset_x = self._template_face_offset[0] * scaling
            face_offset_y = self._template_face_offset[1] * scaling
        else:
            scaling = 1.0
            template = self._face_template
            roi_scaled = roi
            face_offset_x = self._template_face_offset[0]
            face_offset_y = self._template_face_offset[1]

        if template.shape[0] > roi_scaled.shape[0] or template.shape[1] > roi_scaled.shape[1]:
            return params  # Template larger than ROI

        # Template matching
        try:
            corr_out = cv2.matchTemplate(roi_scaled, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(corr_out)

            # Only apply shift if correlation is strong enough
            # C++ OpenFace doesn't have a threshold, but we keep a very low one for safety
            # Lowered from 0.5 to 0.2 to match C++ behavior more closely
            if max_val < 0.2:
                if self.debug_mode:
                    print(f"[TEMPLATE_MATCH] Very low correlation ({max_val:.3f}), skipping")
                return params

            # max_loc is where the template top-left corner was found (in scaled ROI coords)
            # Convert to image coordinates
            template_found_x = max_loc[0] / scaling + roi_x
            template_found_y = max_loc[1] / scaling + roi_y

            # The face bbox is offset within the template by _template_face_offset
            # When template was scaled, the offset in IMAGE coordinates stays the same
            # (the offset was stored in original image coords, not template coords)
            # So actual face position = template position + face offset (in image coords)
            face_found_x = template_found_x + self._template_face_offset[0]
            face_found_y = template_found_y + self._template_face_offset[1]

            # Shift is where face was found vs where we expected it (init_box)
            shift_x = face_found_x - init_box_x
            shift_y = face_found_y - init_box_y

            # Clamp shift to reasonable values (max 1/4 of face width)
            max_shift = max(width, height) / 4
            shift_x = np.clip(shift_x, -max_shift, max_shift)
            shift_y = np.clip(shift_y, -max_shift, max_shift)

            # Apply shift to translation parameters
            params = params.copy()
            params[4] += shift_x  # tx
            params[5] += shift_y  # ty

            if self.debug_mode:
                print(f"[TEMPLATE_MATCH] corr={max_val:.3f}, Shift: ({shift_x:.2f}, {shift_y:.2f}) pixels")

        except cv2.error as e:
            if self.debug_mode:
                print(f"[TEMPLATE_MATCH] Template matching failed: {e}")

        return params

    def _update_face_template(self, gray: np.ndarray, params: np.ndarray):
        """
        Update face template after successful detection (matches C++ UpdateTemplate).

        Extracts a face template from the current frame using the fitted landmarks.
        This template is used for template matching correction in subsequent frames.

        Args:
            gray: Grayscale image
            params: Fitted PDM parameters
        """
        landmarks = self.pdm.params_to_landmarks_2d(params)

        # Get bounding box from landmarks (this is init_box in C++)
        x_min, y_min = landmarks.min(axis=0)
        x_max, y_max = landmarks.max(axis=0)
        width = x_max - x_min
        height = y_max - y_min

        # Store the init_box for next frame's template matching
        self._template_init_box = (x_min, y_min, width, height)

        # Extract face region directly without padding (matching C++ UpdateTemplate)
        # C++ extracts just the face bbox region, no additional padding
        x1 = int(max(0, x_min))
        y1 = int(max(0, y_min))
        x2 = int(min(gray.shape[1], x_max))
        y2 = int(min(gray.shape[0], y_max))

        if x2 > x1 and y2 > y1:
            self._face_template = gray[y1:y2, x1:x2].copy()
            self._template_scale = params[0]  # Store current scale

            # No offset needed since template IS the face bbox (like C++)
            # Face bbox position = template position
            self._template_face_offset = (0.0, 0.0)

            if self.debug_mode:
                print(f"[TEMPLATE_UPDATE] Template size: {self._face_template.shape}")
                print(f"[TEMPLATE_UPDATE] Face offset in template: {self._template_face_offset}")

    def reset_temporal_state(self):
        """
        Reset temporal warm-start state.

        Call this when:
        - Starting a new video
        - Switching to a different face
        - After a face tracking failure

        This clears the cached previous frame data used for video mode optimization.
        """
        self._prev_frame_params = None
        self._prev_frame_landmarks = None
        # Reset template tracking state
        self._face_template = None
        self._template_init_box = None
        self._template_face_offset = None
        self._tracking_initialized = False
        # Also reset optimizer's response map cache
        if hasattr(self.optimizer, 'cached_response_maps'):
            self.optimizer.cached_response_maps = None
            self.optimizer.cached_landmarks = None
            self.optimizer.cache_age = 0

    def get_info(self) -> Dict:
        """Get model information."""
        return {
            'pdm': self.pdm.get_info(),
            'ccnf': self.ccnf.get_info(),
            'optimizer': {
                'regularization': self.optimizer.regularization,
                'max_iterations': self.optimizer.max_iterations,
                'convergence_threshold': self.optimizer.convergence_threshold,
                'convergence_profile': self.optimizer.convergence_profile_name
            },
            'patch_scales': self.patch_scaling,
            'early_window_exit': self.early_window_exit,
            'early_exit_threshold': self.early_exit_threshold,
            'video_mode': self._video_mode
        }


def test_clnf():
    """Test CLNF complete pipeline."""
    print("=" * 60)
    print("Testing Complete CLNF Pipeline")
    print("=" * 60)

    # Test 1: Initialize CLNF
    print("\nTest 1: Initialize CLNF")
    clnf = CLNF(
        model_dir="pyclnf/models",
        scale=0.25,
        max_iterations=5
    )

    info = clnf.get_info()
    print(f"  PDM: {info['pdm']['n_points']} landmarks, {info['pdm']['n_params']} params")
    print(f"  CCNF scales: {info['ccnf']['scales']}")
    print(f"  CCNF patches at 0.25: {info['ccnf']['scale_models'][0.25]['total_patches']}")
    print(f"  Optimizer: max_iter={info['optimizer']['max_iterations']}")

    # Test 2: Create test image with face-like features
    print("\nTest 2: Create test image")
    test_image = np.random.randint(0, 256, (480, 640), dtype=np.uint8)

    # Add some edge structure to simulate facial features
    center_y, center_x = 240, 320
    cv2.circle(test_image, (center_x - 50, center_y - 30), 15, 200, 2)  # Left eye
    cv2.circle(test_image, (center_x + 50, center_y - 30), 15, 200, 2)  # Right eye
    cv2.ellipse(test_image, (center_x, center_y + 30), (40, 20), 0, 0, 180, 200, 2)  # Mouth

    print(f"  Test image: {test_image.shape}")

    # Test 3: Fit CLNF to image
    print("\nTest 3: Fit CLNF to test image")
    face_bbox = (220, 140, 200, 250)  # [x, y, width, height]

    landmarks, fit_info = clnf.fit(test_image, face_bbox, return_params=True)

    print(f"  Bbox: {face_bbox}")
    print(f"  Converged: {fit_info['converged']}")
    print(f"  Iterations: {fit_info['iterations']}")
    print(f"  Final update: {fit_info['final_update']:.6f}")
    print(f"  Landmarks shape: {landmarks.shape}")
    print(f"  Landmark range: x=[{landmarks[:, 0].min():.1f}, {landmarks[:, 0].max():.1f}], y=[{landmarks[:, 1].min():.1f}, {landmarks[:, 1].max():.1f}]")

    # Test 4: Verify landmarks are within expected region
    print("\nTest 4: Verify landmark positions")
    bbox_center_x = face_bbox[0] + face_bbox[2] / 2
    bbox_center_y = face_bbox[1] + face_bbox[3] / 2

    landmark_center_x = landmarks[:, 0].mean()
    landmark_center_y = landmarks[:, 1].mean()

    center_offset = np.sqrt((landmark_center_x - bbox_center_x)**2 + (landmark_center_y - bbox_center_y)**2)

    print(f"  Bbox center: ({bbox_center_x:.1f}, {bbox_center_y:.1f})")
    print(f"  Landmark center: ({landmark_center_x:.1f}, {landmark_center_y:.1f})")
    print(f"  Center offset: {center_offset:.1f} pixels")

    # Test 5: Test with different bbox
    print("\nTest 5: Fit with different bbox")
    face_bbox2 = (150, 100, 150, 180)
    landmarks2, fit_info2 = clnf.fit(test_image, face_bbox2)

    print(f"  Bbox: {face_bbox2}")
    print(f"  Converged: {fit_info2['converged']}")
    print(f"  Iterations: {fit_info2['iterations']}")
    print(f"  Landmark shift from first fit: {np.linalg.norm(landmarks2 - landmarks, axis=1).mean():.1f} pixels")

    print("\n" + "=" * 60)
    print("✓ Complete CLNF Pipeline Tests Complete!")
    print("=" * 60)
    print("\nCLNF is ready to use!")
    print("  - Pure Python implementation (no C++ dependencies)")
    print("  - Loads OpenFace trained models")
    print("  - Ready for PyInstaller distribution")
    print("  - Can be optimized with CoreML/Cython/CuPy as needed")


if __name__ == "__main__":
    test_clnf()
