"""
OpenFace Model Loader - Parse OpenFace models directly without C++ dependency

This module provides utilities to load OpenFace PDM and patch expert models
directly from their native text/binary formats.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import struct


class PDMLoader:
    """Load and parse OpenFace PDM (Point Distribution Model) from text file."""

    def __init__(self, model_path: str):
        """
        Load PDM from OpenFace text format.

        Args:
            model_path: Path to PDM .txt file (e.g., In-the-wild_aligned_PDM_68.txt)
        """
        self.model_path = Path(model_path)
        self.mean_shape = None
        self.princ_comp = None  # Principal components
        self.eigen_values = None

        self._load()

    def _skip_comments(self, f):
        """Skip lines starting with # and empty lines."""
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                return
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                # Not a comment, go back
                f.seek(pos)
                return

    def _read_matrix(self, f) -> np.ndarray:
        """
        Read a matrix in OpenFace text format:
        Line 1: rows
        Line 2: cols
        Line 3: type (OpenCV type code)
        Remaining lines: data values (one row per line, space-separated)
        """
        self._skip_comments(f)
        rows = int(f.readline().strip())

        self._skip_comments(f)
        cols = int(f.readline().strip())

        self._skip_comments(f)
        cv_type = int(f.readline().strip())

        # CV_64FC1 = 6, CV_32FC1 = 5
        # Determine data type from OpenCV type code
        if cv_type == 6:  # CV_64FC1
            dtype = np.float64
        elif cv_type == 5:  # CV_32FC1
            dtype = np.float32
        else:
            dtype = np.float64  # Default to double

        # Read data values (one row per line, space-separated)
        data = []
        for _ in range(rows):
            self._skip_comments(f)
            line = f.readline().strip()
            # Split by whitespace and convert to floats
            row_values = [float(x) for x in line.split()]
            if len(row_values) != cols:
                raise ValueError(f"Expected {cols} values, got {len(row_values)}")
            data.extend(row_values)

        # Reshape to matrix
        matrix = np.array(data, dtype=dtype).reshape(rows, cols)
        return matrix

    def _load(self):
        """Load all PDM components from file."""
        with open(self.model_path, 'r') as f:
            # Read mean shape
            self.mean_shape = self._read_matrix(f)

            # Read principal components
            self.princ_comp = self._read_matrix(f)

            # Read eigenvalues
            self.eigen_values = self._read_matrix(f)

    def save_numpy(self, output_dir: str):
        """
        Export PDM to NumPy .npy files.

        Args:
            output_dir: Directory to save .npy files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        np.save(output_path / 'mean_shape.npy', self.mean_shape.astype(np.float32))
        np.save(output_path / 'princ_comp.npy', self.princ_comp.astype(np.float32))
        np.save(output_path / 'eigen_values.npy', self.eigen_values.astype(np.float32))

        print(f"PDM exported to {output_path}")
        print(f"  - mean_shape.npy: shape {self.mean_shape.shape}")
        print(f"  - princ_comp.npy: shape {self.princ_comp.shape}")
        print(f"  - eigen_values.npy: shape {self.eigen_values.shape}")

    def number_of_points(self) -> int:
        """Return number of landmarks (should be 68 for face)."""
        return self.mean_shape.shape[0] // 3

    def number_of_modes(self) -> int:
        """Return number of PCA modes."""
        return self.princ_comp.shape[1]

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'num_points': self.number_of_points(),
            'num_modes': self.number_of_modes(),
            'mean_shape_shape': self.mean_shape.shape,
            'princ_comp_shape': self.princ_comp.shape,
            'eigen_values_shape': self.eigen_values.shape,
        }


class CCNFPatchExpertLoader:
    """Load and parse OpenFace CCNF patch experts from binary file."""

    def __init__(self, model_path: str, num_landmarks: int = 68):
        """
        Load CCNF patch experts from OpenFace binary format.

        Args:
            model_path: Path to CCNF .txt file (e.g., ccnf_patches_0.25_general.txt)
            num_landmarks: Number of landmarks (default: 68)
        """
        self.model_path = Path(model_path)
        self.num_landmarks = num_landmarks

        # Multi-view structure
        self.patch_scaling = None
        self.num_views = 0
        self.centers = []  # List of (3, 1) arrays - [pitch, yaw, roll] per view
        self.visibilities = []  # List of (num_landmarks, 1) arrays - 0/1 per landmark per view
        self.window_sizes = []  # List of window sizes for edge features
        self.sigma_components = []  # List of lists of sigma matrices
        self.patches = []  # List of lists: patches[view_idx][landmark_idx]

        self._load()

    def _read_int32(self, f) -> int:
        """Read 4-byte integer (little-endian)."""
        return struct.unpack('<i', f.read(4))[0]

    def _read_float64(self, f) -> float:
        """Read 8-byte double (little-endian)."""
        return struct.unpack('<d', f.read(8))[0]

    def _read_matrix_bin(self, f) -> np.ndarray:
        """
        Read binary matrix in OpenFace format:
        - 4 bytes: rows (int)
        - 4 bytes: cols (int)
        - 4 bytes: type (OpenCV type code)
        - remaining: data
        """
        rows = self._read_int32(f)
        cols = self._read_int32(f)
        cv_type = self._read_int32(f)

        # Determine dtype and element size
        if cv_type == 4:  # CV_32SC1 (int32)
            dtype = np.int32
            elem_size = 4
        elif cv_type == 5:  # CV_32FC1 (float32)
            dtype = np.float32
            elem_size = 4
        elif cv_type == 6:  # CV_64FC1 (float64)
            dtype = np.float64
            elem_size = 8
        else:
            raise ValueError(f"Unsupported OpenCV type: {cv_type}")

        # Read data
        num_elements = rows * cols
        data = np.frombuffer(f.read(num_elements * elem_size), dtype=dtype)

        return data.reshape(rows, cols)

    def _read_neuron(self, f) -> Dict[str, Any]:
        """Read a single CCNF neuron from binary file."""
        # Read type marker (should be 2)
        read_type = self._read_int32(f)
        if read_type != 2:
            raise ValueError(f"Expected neuron type 2, got {read_type}")

        # Read neuron parameters
        neuron_type = self._read_int32(f)
        norm_weights = self._read_float64(f)
        bias = self._read_float64(f)
        alpha = self._read_float64(f)

        # Read weight matrix
        weights = self._read_matrix_bin(f)

        return {
            'neuron_type': neuron_type,
            'norm_weights': norm_weights,
            'bias': bias,
            'alpha': alpha,
            'weights': weights
        }

    def _read_patch_expert(self, f, window_sizes: list, n_sigmas: int, debug=False) -> Dict[str, Any]:
        """
        Read a single patch expert (for one landmark).

        Args:
            f: File handle
            window_sizes: List of window sizes for sigma computation
            n_sigmas: Number of sigma components (for beta reading)
            debug: Print debug information
        """
        # Read patch expert header
        read_type = self._read_int32(f)
        if debug:
            print(f"      [DEBUG] read_type = {read_type}")
        if read_type != 5:
            raise ValueError(f"Expected patch expert type 5, got {read_type}")

        width = self._read_int32(f)
        height = self._read_int32(f)
        num_neurons = self._read_int32(f)

        # Handle empty patches (landmark not visible at this orientation)
        if num_neurons == 0:
            # Read empty marker
            self._read_int32(f)
            return {
                'width': width,
                'height': height,
                'neurons': [],
                'betas': [],
                'patch_confidence': 0.0,
                'empty': True
            }

        # Read neurons
        neurons = []
        for _ in range(num_neurons):
            neuron = self._read_neuron(f)
            neurons.append(neuron)

        # Read beta values (for edge features)
        # n_betas = sigma_components[0].size() from C++ code
        n_betas = n_sigmas
        betas = []
        for i in range(n_betas):
            beta = self._read_float64(f)
            betas.append(beta)
            if debug:
                print(f"      [DEBUG] beta[{i}] = {beta}")

        # Read patch confidence
        patch_confidence = self._read_float64(f)
        if debug:
            print(f"      [DEBUG] patch_confidence = {patch_confidence}")

        return {
            'width': width,
            'height': height,
            'neurons': neurons,
            'betas': betas,
            'patch_confidence': patch_confidence,
            'empty': False
        }

    def _load(self):
        """Load CCNF patch experts from binary file with multi-view structure."""
        print(f"Loading CCNF patch experts from {self.model_path}")

        with open(self.model_path, 'rb') as f:
            # Read file header
            self.patch_scaling = self._read_float64(f)
            self.num_views = self._read_int32(f)

            print(f"  Patch scaling: {self.patch_scaling}")
            print(f"  Number of views: {self.num_views}")

            # Read view centers (pitch, yaw, roll for each view)
            print(f"\n  Reading {self.num_views} view centers...")
            for view_idx in range(self.num_views):
                center = self._read_matrix_bin(f)  # Should be (3, 1)
                # Convert from radians to degrees as per C++ code
                center_deg = center * 180.0 / np.pi
                self.centers.append(center_deg)
                print(f"    View {view_idx}: pitch={center_deg[0,0]:.1f}°, "
                      f"yaw={center_deg[1,0]:.1f}°, roll={center_deg[2,0]:.1f}°")

            # Read visibility matrices (which landmarks are visible in each view)
            print(f"\n  Reading {self.num_views} visibility matrices...")
            for view_idx in range(self.num_views):
                visibility = self._read_matrix_bin(f)  # Should be (num_landmarks, 1)
                self.visibilities.append(visibility)
                num_visible = np.sum(visibility)
                print(f"    View {view_idx}: {num_visible}/{self.num_landmarks} landmarks visible")

            # Read window sizes and sigma components (for edge features)
            print(f"\n  Reading window sizes and sigma components...")
            num_win_sizes = self._read_int32(f)
            print(f"    Number of window sizes: {num_win_sizes}")

            for w in range(num_win_sizes):
                window_size = self._read_int32(f)
                num_sigma_comp = self._read_int32(f)
                self.window_sizes.append(window_size)

                sigmas_for_window = []
                for s in range(num_sigma_comp):
                    sigma_mat = self._read_matrix_bin(f)
                    sigmas_for_window.append(sigma_mat)

                self.sigma_components.append(sigmas_for_window)
                print(f"    Window size {window_size}: {num_sigma_comp} sigma components")

            # Read patch experts for each view and landmark
            print(f"\n  Reading patch experts...")
            for view_idx in range(self.num_views):
                view_patches = []
                print(f"\n  View {view_idx}:")

                for landmark_idx in range(self.num_landmarks):
                    try:
                        # Check if this landmark is visible in this view
                        is_visible = self.visibilities[view_idx][landmark_idx, 0] == 1

                        if not is_visible:
                            # Store empty marker
                            view_patches.append({
                                'width': 0,
                                'height': 0,
                                'neurons': [],
                                'betas': [],
                                'patch_confidence': 0.0,
                                'empty': True
                            })
                            continue

                        # Read patch expert
                        n_sigmas = len(self.sigma_components[0]) if self.sigma_components else 0
                        patch = self._read_patch_expert(f, self.window_sizes, n_sigmas, debug=False)
                        view_patches.append(patch)

                        if not patch['empty'] and landmark_idx % 10 == 0:
                            print(f"    Landmark {landmark_idx:02d}: {len(patch['neurons'])} neurons, "
                                  f"{patch['width']}x{patch['height']} patch, "
                                  f"confidence={patch['patch_confidence']:.4f}")

                    except struct.error as e:
                        print(f"    Error reading landmark {landmark_idx} in view {view_idx}: {e}")
                        raise
                    except Exception as e:
                        print(f"    Error reading landmark {landmark_idx} in view {view_idx}: {e}")
                        raise

                self.patches.append(view_patches)

        print(f"\nLoaded {len(self.patches)} views with {self.num_landmarks} patch experts each")

    def save_numpy(self, output_dir: str):
        """Export patch experts to NumPy format with multi-view structure."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting CCNF patches to {output_path}")

        # Save global metadata
        global_meta = output_path / 'global_metadata.npz'
        np.savez(
            global_meta,
            patch_scaling=self.patch_scaling,
            num_views=self.num_views,
            num_landmarks=self.num_landmarks,
            window_sizes=np.array(self.window_sizes)
        )
        print(f"  Saved global metadata: {self.num_views} views, {self.num_landmarks} landmarks")

        # Save sigma components for each window size
        print(f"  Saving {len(self.window_sizes)} window size sigma components...")
        for w_idx, window_size in enumerate(self.window_sizes):
            sigma_dir = output_path / f'sigmas_window_{window_size}'
            sigma_dir.mkdir(exist_ok=True)

            sigmas_for_window = self.sigma_components[w_idx]
            for s_idx, sigma_mat in enumerate(sigmas_for_window):
                sigma_file = sigma_dir / f'sigma_{s_idx:02d}.npy'
                np.save(sigma_file, sigma_mat.astype(np.float32))

            print(f"    Window {window_size}: Saved {len(sigmas_for_window)} sigma matrices")

        # Save view centers and visibilities
        for view_idx in range(self.num_views):
            view_meta_file = output_path / f'view_{view_idx:02d}_metadata.npz'
            np.savez(
                view_meta_file,
                center=self.centers[view_idx],
                visibility=self.visibilities[view_idx]
            )

        # Export each view's patch experts
        for view_idx, view_patches in enumerate(self.patches):
            view_dir = output_path / f'view_{view_idx:02d}'
            view_dir.mkdir(exist_ok=True)

            non_empty_count = 0
            for landmark_idx, patch in enumerate(view_patches):
                if patch.get('empty', False):
                    # Create empty marker file
                    marker_file = view_dir / f'patch_{landmark_idx:02d}_empty.txt'
                    marker_file.write_text('empty')
                    continue

                non_empty_count += 1
                patch_dir = view_dir / f'patch_{landmark_idx:02d}'
                patch_dir.mkdir(exist_ok=True)

                # Save patch metadata
                metadata_file = patch_dir / 'metadata.npz'
                np.savez(
                    metadata_file,
                    width=patch['width'],
                    height=patch['height'],
                    betas=np.array(patch['betas']),
                    patch_confidence=patch['patch_confidence']
                )

                # Save neurons
                for j, neuron in enumerate(patch['neurons']):
                    neuron_file = patch_dir / f'neuron_{j:02d}.npz'
                    np.savez(
                        neuron_file,
                        neuron_type=neuron['neuron_type'],
                        norm_weights=neuron['norm_weights'],
                        bias=neuron['bias'],
                        alpha=neuron['alpha'],
                        weights=neuron['weights']
                    )

            print(f"  View {view_idx}: Exported {non_empty_count}/{self.num_landmarks} non-empty patches")

        print(f"\n✓ Export complete: {self.num_views} views × {self.num_landmarks} patches")

    def get_info(self) -> Dict[str, Any]:
        """Get patch expert collection information for multi-view structure."""
        if not self.patches:
            return {
                'patch_scaling': self.patch_scaling,
                'num_views': self.num_views,
                'num_landmarks': self.num_landmarks,
                'num_patches': 0
            }

        # Collect all non-empty patches across all views
        all_non_empty = []
        view_stats = []

        for view_idx, view_patches in enumerate(self.patches):
            non_empty = [p for p in view_patches if not p.get('empty', False)]
            all_non_empty.extend(non_empty)

            view_stats.append({
                'view': view_idx,
                'center': self.centers[view_idx].flatten().tolist(),
                'num_visible': len(non_empty),
                'num_invisible': sum(1 for p in view_patches if p.get('empty', False))
            })

        if not all_non_empty:
            return {
                'patch_scaling': self.patch_scaling,
                'num_views': self.num_views,
                'num_landmarks': self.num_landmarks,
                'view_stats': view_stats
            }

        neuron_counts = [len(p['neurons']) for p in all_non_empty]

        return {
            'patch_scaling': self.patch_scaling,
            'num_views': self.num_views,
            'num_landmarks': self.num_landmarks,
            'total_patches': len(self.patches) * self.num_landmarks,
            'total_non_empty': len(all_non_empty),
            'avg_neurons_per_patch': np.mean(neuron_counts),
            'min_neurons': np.min(neuron_counts),
            'max_neurons': np.max(neuron_counts),
            'patch_sizes': [(p['width'], p['height']) for p in all_non_empty[:5]],
            'view_stats': view_stats
        }


def test_pdm_loader():
    """Test PDM loading functionality."""
    openface_dir = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace"
    pdm_path = openface_dir / "lib/local/LandmarkDetector/model/pdms/In-the-wild_aligned_PDM_68.txt"

    if not pdm_path.exists():
        print(f"PDM file not found: {pdm_path}")
        return

    print("=" * 60)
    print("Testing PDM Loader")
    print("=" * 60)
    pdm = PDMLoader(str(pdm_path))

    info = pdm.get_info()
    print("\nPDM Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Export to numpy
    output_dir = "pyclnf/models/exported_pdm"
    pdm.save_numpy(output_dir)

    # Verify by loading
    print("\nVerifying exported files...")
    mean_shape = np.load(f"{output_dir}/mean_shape.npy")
    princ_comp = np.load(f"{output_dir}/princ_comp.npy")
    eigen_values = np.load(f"{output_dir}/eigen_values.npy")

    print(f"  Loaded mean_shape: {mean_shape.shape}, dtype: {mean_shape.dtype}")
    print(f"  Loaded princ_comp: {princ_comp.shape}, dtype: {princ_comp.dtype}")
    print(f"  Loaded eigen_values: {eigen_values.shape}, dtype: {eigen_values.dtype}")

    print("\n✓ PDM loading and export successful!")
    return True


def test_ccnf_loader():
    """Test CCNF patch expert loading functionality."""
    openface_dir = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace"
    ccnf_path = openface_dir / "lib/local/LandmarkDetector/model/patch_experts/ccnf_patches_0.25_general.txt"

    if not ccnf_path.exists():
        print(f"CCNF file not found: {ccnf_path}")
        return False

    print("\n" + "=" * 60)
    print("Testing CCNF Patch Expert Loader")
    print("=" * 60)

    try:
        ccnf = CCNFPatchExpertLoader(str(ccnf_path), num_landmarks=68)

        info = ccnf.get_info()
        print("\nCCNF Patch Expert Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Export to numpy
        output_dir = "pyclnf/models/exported_ccnf_0.25"
        ccnf.save_numpy(output_dir)

        # Verify by loading one patch (view 0, landmark 0)
        print("\nVerifying exported files...")
        if ccnf.patches and ccnf.patches[0] and not ccnf.patches[0][0].get('empty', False):
            patch_meta = np.load(f"{output_dir}/view_00/patch_00/metadata.npz")
            patch_neuron = np.load(f"{output_dir}/view_00/patch_00/neuron_00.npz")

            print(f"  View 0, Patch 0 metadata keys: {list(patch_meta.keys())}")
            print(f"  View 0, Patch 0, Neuron 0 keys: {list(patch_neuron.keys())}")
            print(f"  Neuron weights shape: {patch_neuron['weights'].shape}")

        print("\n✓ CCNF loading and export successful!")
        return True

    except Exception as e:
        print(f"\n✗ CCNF loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_sigma_components(model_dir: str) -> Optional[Dict[int, np.ndarray]]:
    """
    Load CCNF sigma components for spatial correlation modeling.

    Sigma components are used to compute the Sigma covariance matrix that
    transforms response maps to model spatial correlations.

    Args:
        model_dir: Base model directory containing sigma_components subdirectory

    Returns:
        Dictionary mapping window_size -> list of sigma component matrices
        Returns None if sigma components directory doesn't exist
    """
    sigma_dir = Path(model_dir) / "sigma_components"
    if not sigma_dir.exists():
        return None

    # Load window sizes
    window_sizes_file = sigma_dir / "window_sizes.npy"
    if not window_sizes_file.exists():
        return None

    window_sizes = np.load(window_sizes_file)

    # Load sigma components for each window size
    sigma_components = {}
    for window_size in window_sizes:
        components = []
        component_idx = 0
        while True:
            sigma_file = sigma_dir / f"sigma_w{window_size}_c{component_idx}.npy"
            if not sigma_file.exists():
                break
            components.append(np.load(sigma_file))
            component_idx += 1

        if components:
            sigma_components[int(window_size)] = components

    return sigma_components


if __name__ == "__main__":
    # Test PDM
    pdm_success = test_pdm_loader()

    # Test CCNF
    ccnf_success = test_ccnf_loader()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"PDM:  {'✓ PASS' if pdm_success else '✗ FAIL'}")
    print(f"CCNF: {'✓ PASS' if ccnf_success else '✗ FAIL'}")
