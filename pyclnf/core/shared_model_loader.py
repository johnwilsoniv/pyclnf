#!/usr/bin/env python3
"""
Shared Memory Model Loader for HPC environments.

Enables multiple worker processes to share a single copy of CEN patch expert
models in memory, reducing RAM usage from 424MB × N_workers to 424MB × 1.

Usage:
    # Manager process (run once before spawning workers):
    SharedModelManager.initialize('/dev/shm/pyclnf_models', model_dir)

    # Worker processes:
    experts = SharedCENPatchExperts('/dev/shm/pyclnf_models')

    # Cleanup when done:
    SharedModelManager.cleanup('/dev/shm/pyclnf_models')
"""

import numpy as np
from pathlib import Path
import os
import json
import struct
import filelock
from typing import Dict, List, Optional, Tuple


class SharedModelManager:
    """
    Manages creation and cleanup of shared memory model files.

    Should be called once by the main process before spawning workers.
    """

    @staticmethod
    def initialize(shm_dir: str, model_dir: str, force: bool = False) -> bool:
        """
        Load models and write to shared memory directory.

        Args:
            shm_dir: Directory for shared memory files (e.g., '/dev/shm/pyclnf_models')
            model_dir: Source directory containing CEN .dat files
            force: If True, recreate even if already exists

        Returns:
            True if models were created, False if already existed
        """
        shm_path = Path(shm_dir)
        model_path = Path(model_dir)

        # Create lock to prevent race conditions
        lock_file = shm_path.parent / f"{shm_path.name}.lock"
        lock = filelock.FileLock(str(lock_file), timeout=60)

        with lock:
            # Check if already initialized
            manifest_file = shm_path / "manifest.json"
            if manifest_file.exists() and not force:
                try:
                    with open(manifest_file, 'r') as f:
                        manifest = json.load(f)
                    if manifest.get('complete', False):
                        print(f"[SharedModelManager] Models already loaded at {shm_dir}")
                        return False
                except:
                    pass  # Corrupted, recreate

            # Create directory
            shm_path.mkdir(parents=True, exist_ok=True)

            print(f"[SharedModelManager] Initializing shared models at {shm_dir}...")

            # Import the regular CEN loader to read the .dat files
            from pyclnf.core.cen_patch_expert import CENPatchExperts

            # Load models using regular loader
            experts = CENPatchExperts(model_dir)

            # Serialize to shared memory format
            manifest = {
                'model_dir': str(model_path),
                'patch_scaling': experts.patch_scaling,
                'num_landmarks': experts.num_landmarks,
                'num_scales': len(experts.patch_experts),
                'scales': [],
                'complete': False
            }

            # Save mirror indices
            np.save(shm_path / "mirror_inds.npy", experts.mirror_inds)

            # Save each scale's experts
            for scale_idx, scale_experts in enumerate(experts.patch_experts):
                scale_value = experts.patch_scaling[scale_idx]
                scale_dir = shm_path / f"scale_{scale_idx}"
                scale_dir.mkdir(exist_ok=True)

                scale_info = {
                    'scale_value': scale_value,
                    'num_experts': len(scale_experts),
                    'experts': []
                }

                for lm_idx, expert in enumerate(scale_experts):
                    expert_info = SharedModelManager._save_expert(
                        expert, scale_dir, lm_idx
                    )
                    scale_info['experts'].append(expert_info)

                manifest['scales'].append(scale_info)
                print(f"  Scale {scale_value}: {len(scale_experts)} experts serialized")

            # Mark as complete
            manifest['complete'] = True
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)

            print(f"[SharedModelManager] Shared models initialized successfully")
            return True

    @staticmethod
    def _save_expert(expert, scale_dir: Path, lm_idx: int) -> dict:
        """Save a single CEN patch expert to disk."""
        expert_info = {
            'landmark_idx': lm_idx,
            'width_support': expert.width_support,
            'height_support': expert.height_support,
            'confidence': expert.confidence,
            'is_empty': expert.is_empty,
            'activation_function': expert.activation_function,
            'num_layers': len(expert.weights)
        }

        if not expert.is_empty:
            # Save weights and biases as memory-mappable numpy files
            for layer_idx, (weight, bias) in enumerate(zip(expert.weights, expert.biases)):
                weight_file = scale_dir / f"lm{lm_idx:02d}_l{layer_idx}_weight.npy"
                bias_file = scale_dir / f"lm{lm_idx:02d}_l{layer_idx}_bias.npy"
                # Convert to float32 for memory efficiency
                np.save(weight_file, weight.astype(np.float32))
                np.save(bias_file, bias.astype(np.float32))

        return expert_info

    @staticmethod
    def cleanup(shm_dir: str):
        """Remove shared memory model files."""
        import shutil
        shm_path = Path(shm_dir)
        if shm_path.exists():
            shutil.rmtree(shm_path)
            print(f"[SharedModelManager] Cleaned up {shm_dir}")

        # Also remove lock file
        lock_file = shm_path.parent / f"{shm_path.name}.lock"
        if lock_file.exists():
            lock_file.unlink()

    @staticmethod
    def is_initialized(shm_dir: str) -> bool:
        """Check if shared models are already initialized."""
        manifest_file = Path(shm_dir) / "manifest.json"
        if not manifest_file.exists():
            return False
        try:
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            return manifest.get('complete', False)
        except:
            return False


class SharedCENPatchExpert:
    """
    CEN patch expert that uses memory-mapped weights from shared memory.

    Compatible interface with regular CENPatchExpert but reads weights
    via mmap for zero-copy sharing across processes.
    """

    def __init__(self, expert_info: dict, scale_dir: Path, use_mmap: bool = True):
        """
        Initialize from shared memory files.

        Args:
            expert_info: Metadata dict from manifest
            scale_dir: Directory containing the .npy files
            use_mmap: If True, use memory mapping (recommended for HPC)
        """
        self.width_support = expert_info['width_support']
        self.height_support = expert_info['height_support']
        self.confidence = expert_info['confidence']
        self.is_empty = expert_info['is_empty']
        self.activation_function = expert_info['activation_function']

        self.weights = []
        self.biases = []

        if not self.is_empty:
            lm_idx = expert_info['landmark_idx']
            num_layers = expert_info['num_layers']
            mmap_mode = 'r' if use_mmap else None

            for layer_idx in range(num_layers):
                weight_file = scale_dir / f"lm{lm_idx:02d}_l{layer_idx}_weight.npy"
                bias_file = scale_dir / f"lm{lm_idx:02d}_l{layer_idx}_bias.npy"

                # Memory-map for zero-copy access across processes
                weight = np.load(weight_file, mmap_mode=mmap_mode)
                bias = np.load(bias_file, mmap_mode=mmap_mode)

                self.weights.append(weight)
                self.biases.append(bias)

    @property
    def width(self):
        """Compatibility property for CCNF interface"""
        return self.width_support

    @property
    def height(self):
        """Compatibility property for CCNF interface"""
        return self.height_support

    @property
    def patch_confidence(self):
        """Compatibility property for CCNF interface"""
        return self.confidence

    def compute_sigma(self, sigma_components, window_size=None, debug=False):
        """Return identity matrix for CEN in quick mode."""
        if window_size is None:
            window_size = self.width_support
        matrix_size = window_size * window_size
        return np.eye(matrix_size, dtype=np.float32)

    def response(self, area_of_interest):
        """
        Compute patch expert response map for an image patch.

        Identical to CENPatchExpert.response() - delegates to the same
        computation but uses memory-mapped weights.
        """
        if self.is_empty:
            response_height = max(1, area_of_interest.shape[0] - self.height_support + 1)
            response_width = max(1, area_of_interest.shape[1] - self.width_support + 1)
            return np.zeros((response_height, response_width), dtype=np.float32)

        response_height = area_of_interest.shape[0] - self.height_support + 1
        response_width = area_of_interest.shape[1] - self.width_support + 1

        # Import the optimized response computation
        from pyclnf.core.cen_patch_expert import (
            _response_core_numba, im2col_bias, contrast_norm,
            NUMBA_AVAILABLE
        )

        if NUMBA_AVAILABLE and len(self.weights) == 2:
            input_patch = np.ascontiguousarray(area_of_interest, dtype=np.float32)

            return _response_core_numba(
                input_patch,
                self.width_support,
                self.height_support,
                self.weights[0],
                self.biases[0],
                self.activation_function[0],
                self.weights[1],
                self.biases[1],
                self.activation_function[1],
                response_height,
                response_width
            )
        else:
            # Fallback to regular numpy implementation
            return self._response_numpy(area_of_interest, response_height, response_width)

    def _response_numpy(self, area_of_interest, response_height, response_width):
        """NumPy fallback for response computation."""
        from pyclnf.core.cen_patch_expert import im2col_bias, contrast_norm

        # im2col extraction with bias
        input_col = im2col_bias(area_of_interest, self.width_support, self.height_support)

        # Normalize
        normalized = contrast_norm(input_col)

        # Forward pass through neural network layers
        layer_output = normalized
        for i in range(len(self.weights)):
            layer_output = layer_output @ self.weights[i].T + self.biases[i]

            if self.activation_function[i] == 0:  # Sigmoid
                layer_output = np.clip(layer_output, -88, 88)
                layer_output = 1.0 / (1.0 + np.exp(-layer_output))
            elif self.activation_function[i] == 1:  # Tanh
                layer_output = np.tanh(layer_output)
            elif self.activation_function[i] == 2:  # ReLU
                layer_output = np.maximum(0, layer_output)

        # Reshape in column-major order
        layer_output_flat = layer_output.flatten()
        response = layer_output_flat.reshape(response_height, response_width, order='F')
        return response.astype(np.float32)


class SharedCENPatchExperts:
    """
    Collection of CEN patch experts loaded from shared memory.

    Drop-in replacement for CENPatchExperts that uses memory-mapped
    files for zero-copy sharing across worker processes.
    """

    def __init__(self, shm_dir: str, use_mmap: bool = True):
        """
        Load CEN patch experts from shared memory directory.

        Args:
            shm_dir: Path to shared memory directory (e.g., '/dev/shm/pyclnf_models')
            use_mmap: If True, use memory mapping for weights
        """
        shm_path = Path(shm_dir)
        manifest_file = shm_path / "manifest.json"

        if not manifest_file.exists():
            raise FileNotFoundError(
                f"Shared models not initialized. Run SharedModelManager.initialize() first. "
                f"Missing: {manifest_file}"
            )

        with open(manifest_file, 'r') as f:
            manifest = json.load(f)

        if not manifest.get('complete', False):
            raise RuntimeError("Shared models initialization incomplete")

        self.patch_scaling = manifest['patch_scaling']
        self.num_landmarks = manifest['num_landmarks']
        self.patch_experts = []

        # Load mirror indices (small, just load directly)
        self.mirror_inds = np.load(shm_path / "mirror_inds.npy")

        # Load each scale
        for scale_idx, scale_info in enumerate(manifest['scales']):
            scale_dir = shm_path / f"scale_{scale_idx}"
            experts_at_scale = []

            for expert_info in scale_info['experts']:
                expert = SharedCENPatchExpert(expert_info, scale_dir, use_mmap=use_mmap)
                experts_at_scale.append(expert)

            self.patch_experts.append(experts_at_scale)

    def get_expert(self, scale_idx: int, landmark_idx: int):
        """Get patch expert for a specific scale and landmark."""
        return self.patch_experts[scale_idx][landmark_idx]

    def get_experts_at_scale(self, scale_idx: int) -> List:
        """Get all patch experts at a specific scale."""
        return self.patch_experts[scale_idx]


def get_cen_patch_experts(model_dir: str, shm_dir: Optional[str] = None,
                          use_shared: bool = True) -> 'CENPatchExperts':
    """
    Factory function to get CEN patch experts with optional shared memory.

    Args:
        model_dir: Path to model directory containing .dat files
        shm_dir: Path for shared memory (default: /dev/shm/pyclnf_models)
        use_shared: If True, use shared memory (recommended for multiprocessing)

    Returns:
        CENPatchExperts or SharedCENPatchExperts instance
    """
    if not use_shared:
        from pyclnf.core.cen_patch_expert import CENPatchExperts
        return CENPatchExperts(model_dir)

    if shm_dir is None:
        shm_dir = "/dev/shm/pyclnf_models"

    # Ensure initialized
    if not SharedModelManager.is_initialized(shm_dir):
        SharedModelManager.initialize(shm_dir, model_dir)

    return SharedCENPatchExperts(shm_dir)


# ============================================================================
# AU SVR Model Shared Memory Support
# ============================================================================

class SharedAUModelManager:
    """
    Manages creation and cleanup of shared memory AU SVR model files.

    The AU SVR models are smaller (~10MB total) but still benefit from
    memory-mapping when running many workers.
    """

    @staticmethod
    def initialize(shm_dir: str, au_models_dir: str, force: bool = False) -> bool:
        """
        Load AU SVR models and write to shared memory directory.

        Args:
            shm_dir: Directory for shared memory files (e.g., '/dev/shm/pyfaceau_models')
            au_models_dir: Source directory containing AU_predictors
            force: If True, recreate even if already exists

        Returns:
            True if models were created, False if already existed
        """
        shm_path = Path(shm_dir)

        # Create lock to prevent race conditions
        lock_file = shm_path.parent / f"{shm_path.name}.lock"
        lock = filelock.FileLock(str(lock_file), timeout=60)

        with lock:
            # Check if already initialized
            manifest_file = shm_path / "au_manifest.json"
            if manifest_file.exists() and not force:
                try:
                    with open(manifest_file, 'r') as f:
                        manifest = json.load(f)
                    if manifest.get('complete', False):
                        print(f"[SharedAUModelManager] AU models already loaded at {shm_dir}")
                        return False
                except:
                    pass  # Corrupted, recreate

            # Create directory
            shm_path.mkdir(parents=True, exist_ok=True)

            print(f"[SharedAUModelManager] Initializing shared AU models at {shm_dir}...")

            # Import the regular AU model parser
            import sys
            # Add pyfaceau to path if needed
            pyfaceau_path = Path(au_models_dir).parent.parent
            if str(pyfaceau_path) not in sys.path:
                sys.path.insert(0, str(pyfaceau_path))

            from pyfaceau.prediction.model_parser import OF22ModelParser

            # Load models using regular parser
            parser = OF22ModelParser(au_models_dir)
            models = parser.load_all_models(use_recommended=True, use_combined=True, verbose=False)

            # Serialize to shared memory format
            manifest = {
                'au_models_dir': str(au_models_dir),
                'models': {},
                'complete': False
            }

            au_dir = shm_path / "au_models"
            au_dir.mkdir(exist_ok=True)

            for au_name, model in models.items():
                # Save model arrays as memory-mappable numpy files
                # Use float32 to reduce memory (Phase 4 optimization included here)
                means_file = au_dir / f"{au_name}_means.npy"
                sv_file = au_dir / f"{au_name}_support_vectors.npy"

                np.save(means_file, model['means'].astype(np.float32))
                np.save(sv_file, model['support_vectors'].astype(np.float32))

                manifest['models'][au_name] = {
                    'cutoff': model['cutoff'],
                    'bias': model['bias'],
                    'model_type': model.get('model_type', 'unknown'),
                    'means_shape': list(model['means'].shape),
                    'sv_shape': list(model['support_vectors'].shape)
                }

            print(f"  Serialized {len(models)} AU models")

            # Mark as complete
            manifest['complete'] = True
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)

            print(f"[SharedAUModelManager] Shared AU models initialized successfully")
            return True

    @staticmethod
    def cleanup(shm_dir: str):
        """Remove shared memory AU model files."""
        import shutil
        shm_path = Path(shm_dir)
        au_dir = shm_path / "au_models"
        if au_dir.exists():
            shutil.rmtree(au_dir)
        manifest_file = shm_path / "au_manifest.json"
        if manifest_file.exists():
            manifest_file.unlink()
        print(f"[SharedAUModelManager] Cleaned up AU models at {shm_dir}")

    @staticmethod
    def is_initialized(shm_dir: str) -> bool:
        """Check if shared AU models are already initialized."""
        manifest_file = Path(shm_dir) / "au_manifest.json"
        if not manifest_file.exists():
            return False
        try:
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            return manifest.get('complete', False)
        except:
            return False


class SharedAUModels:
    """
    AU SVR models loaded from shared memory with memory-mapping.

    Provides the same interface as the dict returned by OF22ModelParser.load_all_models()
    but uses memory-mapped arrays for zero-copy sharing.
    """

    def __init__(self, shm_dir: str, use_mmap: bool = True):
        """
        Load AU models from shared memory directory.

        Args:
            shm_dir: Path to shared memory directory
            use_mmap: If True, use memory mapping for arrays
        """
        shm_path = Path(shm_dir)
        manifest_file = shm_path / "au_manifest.json"

        if not manifest_file.exists():
            raise FileNotFoundError(
                f"Shared AU models not initialized. Run SharedAUModelManager.initialize() first."
            )

        with open(manifest_file, 'r') as f:
            manifest = json.load(f)

        if not manifest.get('complete', False):
            raise RuntimeError("Shared AU models initialization incomplete")

        self.models = {}
        au_dir = shm_path / "au_models"
        mmap_mode = 'r' if use_mmap else None

        for au_name, model_info in manifest['models'].items():
            means_file = au_dir / f"{au_name}_means.npy"
            sv_file = au_dir / f"{au_name}_support_vectors.npy"

            self.models[au_name] = {
                'cutoff': model_info['cutoff'],
                'bias': model_info['bias'],
                'model_type': model_info['model_type'],
                'means': np.load(means_file, mmap_mode=mmap_mode),
                'support_vectors': np.load(sv_file, mmap_mode=mmap_mode)
            }

    def __getitem__(self, key):
        return self.models[key]

    def __iter__(self):
        return iter(self.models)

    def __len__(self):
        return len(self.models)

    def items(self):
        return self.models.items()

    def keys(self):
        return self.models.keys()

    def values(self):
        return self.models.values()


def get_au_models(au_models_dir: str, shm_dir: Optional[str] = None,
                  use_shared: bool = True) -> Dict:
    """
    Factory function to get AU SVR models with optional shared memory.

    Args:
        au_models_dir: Path to AU_predictors directory
        shm_dir: Path for shared memory (default: /dev/shm/pyfaceau_models)
        use_shared: If True, use shared memory (recommended for multiprocessing)

    Returns:
        Dict-like object mapping AU names to model dicts
    """
    if not use_shared:
        import sys
        pyfaceau_path = Path(au_models_dir).parent.parent
        if str(pyfaceau_path) not in sys.path:
            sys.path.insert(0, str(pyfaceau_path))
        from pyfaceau.prediction.model_parser import OF22ModelParser
        parser = OF22ModelParser(au_models_dir)
        return parser.load_all_models(use_recommended=True, use_combined=True)

    if shm_dir is None:
        shm_dir = "/dev/shm/pyfaceau_models"

    # Ensure initialized
    if not SharedAUModelManager.is_initialized(shm_dir):
        SharedAUModelManager.initialize(shm_dir, au_models_dir)

    return SharedAUModels(shm_dir)
