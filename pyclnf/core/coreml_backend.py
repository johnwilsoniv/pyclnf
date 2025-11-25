#!/usr/bin/env python3
"""
CoreML Backend for CCNF Patch Experts

Converts CCNF patch expert inference to CoreML for Apple Neural Engine acceleration.
Targets 10-50x speedup over CPU computation.

The CCNF neuron response formula:
    response = (2 * alpha) * sigmoid(correlation * norm_weights + bias)

Where correlation is normalized cross-correlation between the image patch and neuron weights.

This can be implemented as:
1. Depthwise convolution (correlation with weights)
2. Normalization layer (NCC normalization)
3. Scale and bias layer (apply norm_weights and bias)
4. Sigmoid activation
5. Scale by 2*alpha
6. Sum across neurons
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import time


class CoreMLPatchExpert:
    """CoreML-accelerated CCNF patch expert."""

    def __init__(self, patch_dir: str):
        """
        Load and convert patch expert to CoreML.

        Args:
            patch_dir: Directory containing the patch expert data
        """
        self.patch_dir = Path(patch_dir)
        self.coreml_model = None
        self._load_and_convert()

    def _load_and_convert(self):
        """Load patch expert data and convert to CoreML model."""
        try:
            import coremltools as ct
        except ImportError:
            print("Warning: coremltools not available, CoreML backend disabled")
            return

        # Load patch metadata
        meta = np.load(self.patch_dir / 'metadata.npz')
        self.width = int(meta['width'])
        self.height = int(meta['height'])
        self.betas = meta['betas']
        self.patch_confidence = float(meta['patch_confidence'])

        # Load neurons
        neuron_files = sorted(self.patch_dir.glob('neuron_*.npz'))
        self.num_neurons = len(neuron_files)

        if self.num_neurons == 0:
            return

        # Collect neuron parameters
        weights_list = []
        biases = []
        alphas = []
        norm_weights = []

        for neuron_file in neuron_files:
            neuron_data = np.load(neuron_file)
            alpha = float(neuron_data['alpha'])

            # Skip neurons with very small alpha
            if abs(alpha) < 1e-4:
                continue

            weights_list.append(neuron_data['weights'].astype(np.float32))
            biases.append(float(neuron_data['bias']))
            alphas.append(alpha)
            norm_weights.append(float(neuron_data['norm_weights']))

        self.active_neurons = len(weights_list)
        if self.active_neurons == 0:
            return

        # Convert to arrays
        self.weights = np.stack(weights_list, axis=0)  # (n_neurons, h, w)
        self.biases = np.array(biases, dtype=np.float32)
        self.alphas = np.array(alphas, dtype=np.float32)
        self.norm_weights = np.array(norm_weights, dtype=np.float32)

        # Build CoreML model
        self._build_coreml_model(ct)

    def _build_coreml_model(self, ct):
        """Build CoreML model for patch expert inference."""
        try:
            from coremltools.models.neural_network import NeuralNetworkBuilder
            from coremltools.models import MLModel
            from coremltools.models.neural_network.builder import _get_nn_spec

            # Input shape: (1, height, width) - grayscale normalized patch
            input_features = [
                ('input_patch', ct.models.datatypes.Array(1, self.height, self.width))
            ]
            output_features = [
                ('response', ct.models.datatypes.Array(1))
            ]

            # Create neural network builder
            builder = NeuralNetworkBuilder(
                input_features,
                output_features,
                disable_rank5_shape_mapping=True
            )

            # Layer 1: Expand dimensions for convolution (1, h, w) -> (1, 1, h, w)
            builder.add_expand_dims(
                name='expand_dims',
                input_name='input_patch',
                output_name='expanded_input',
                axes=[0]
            )

            # We'll compute NCC in a simplified way for CoreML
            # Use template matching approximation with convolution

            # For each neuron, we need to compute:
            # 1. Subtract mean from patch
            # 2. Convolve with mean-centered weights
            # 3. Divide by norms

            # Build composite model for all neurons
            neuron_outputs = []

            for i in range(self.active_neurons):
                neuron_name = f'neuron_{i}'

                # Get neuron weights and precompute values
                w = self.weights[i]  # (h, w)
                w_mean = np.mean(w)
                w_centered = w - w_mean
                w_norm = np.linalg.norm(w_centered)

                # Reshape weights for conv: (out_channels=1, in_channels=1, h, w)
                conv_weights = w_centered.reshape(1, 1, self.height, self.width)

                # Add convolution layer (computes dot product with centered weights)
                builder.add_convolution(
                    name=f'{neuron_name}_conv',
                    kernel_channels=1,
                    output_channels=1,
                    height=self.height,
                    width=self.width,
                    stride_height=1,
                    stride_width=1,
                    border_mode='valid',
                    groups=1,
                    W=conv_weights.astype(np.float32),
                    b=None,
                    has_bias=False,
                    input_name='expanded_input',
                    output_name=f'{neuron_name}_dot'
                )

                # Apply scale by norm_weights/w_norm and add bias
                # Combined scaling factor
                scale = self.norm_weights[i] / (w_norm + 1e-10)

                builder.add_scale(
                    name=f'{neuron_name}_scale',
                    W=np.array([scale], dtype=np.float32),
                    b=np.array([self.biases[i]], dtype=np.float32),
                    has_bias=True,
                    input_name=f'{neuron_name}_dot',
                    output_name=f'{neuron_name}_scaled'
                )

                # Sigmoid activation
                builder.add_activation(
                    name=f'{neuron_name}_sigmoid',
                    non_linearity='SIGMOID',
                    input_name=f'{neuron_name}_scaled',
                    output_name=f'{neuron_name}_sigmoid_out'
                )

                # Scale by 2*alpha
                builder.add_scale(
                    name=f'{neuron_name}_alpha',
                    W=np.array([2.0 * self.alphas[i]], dtype=np.float32),
                    b=None,
                    has_bias=False,
                    input_name=f'{neuron_name}_sigmoid_out',
                    output_name=f'{neuron_name}_response'
                )

                neuron_outputs.append(f'{neuron_name}_response')

            # Sum all neuron responses (chain pairwise additions)
            if len(neuron_outputs) > 1:
                # Chain additions in pairs since add_broadcastable only takes 2 inputs
                current_sum = neuron_outputs[0]
                for i in range(1, len(neuron_outputs)):
                    output_name = f'sum_{i}' if i < len(neuron_outputs) - 1 else 'total_response'
                    builder.add_add_broadcastable(
                        name=f'add_{i}',
                        input_names=[current_sum, neuron_outputs[i]],
                        output_name=output_name
                    )
                    current_sum = output_name
            else:
                builder.add_activation(
                    name='identity',
                    non_linearity='LINEAR',
                    params=[1.0, 0.0],
                    input_name=neuron_outputs[0],
                    output_name='total_response'
                )

            # Flatten to output shape
            builder.add_flatten_to_2d(
                name='flatten',
                input_name='total_response',
                output_name='response'
            )

            # Build model
            spec = builder.spec
            self.coreml_model = MLModel(spec)

            print(f"CoreML model built for patch {self.patch_dir.name} "
                  f"with {self.active_neurons} neurons")

        except Exception as e:
            print(f"Failed to build CoreML model: {e}")
            self.coreml_model = None

    def compute_response(self, image_patch: np.ndarray) -> float:
        """
        Compute response using CoreML.

        Args:
            image_patch: Grayscale image patch (height, width), values 0-255

        Returns:
            response: Scalar response value
        """
        if self.coreml_model is None:
            # Fallback to CPU computation
            return self._compute_response_cpu(image_patch)

        # Normalize patch to [0, 1]
        patch_normalized = image_patch.astype(np.float32) / 255.0

        # Reshape for CoreML input: (1, height, width)
        input_data = patch_normalized.reshape(1, self.height, self.width)

        # Run inference
        try:
            result = self.coreml_model.predict({'input_patch': input_data})
            return float(result['response'][0])
        except Exception as e:
            print(f"CoreML inference failed: {e}")
            return self._compute_response_cpu(image_patch)

    def _compute_response_cpu(self, image_patch: np.ndarray) -> float:
        """Fallback CPU computation matching the original implementation."""
        # Normalize patch
        features = image_patch.astype(np.float32) / 255.0

        total_response = 0.0
        for i in range(self.active_neurons):
            weights = self.weights[i]

            # Compute NCC
            weight_mean = np.mean(weights)
            feature_mean = np.mean(features)

            weights_centered = weights - weight_mean
            features_centered = features - feature_mean

            weight_norm = np.linalg.norm(weights_centered)
            feature_norm = np.linalg.norm(features_centered)

            if weight_norm > 1e-10 and feature_norm > 1e-10:
                correlation = np.sum(weights_centered * features_centered) / (weight_norm * feature_norm)
            else:
                correlation = 0.0

            # Apply formula
            sigmoid_input = correlation * self.norm_weights[i] + self.biases[i]
            if sigmoid_input >= 0:
                sigmoid_val = 1.0 / (1.0 + np.exp(-sigmoid_input))
            else:
                exp_x = np.exp(sigmoid_input)
                sigmoid_val = exp_x / (1.0 + exp_x)

            response = 2.0 * self.alphas[i] * sigmoid_val
            total_response += response

        return float(total_response)


def convert_all_patch_experts(model_dir: str, scale: float = 0.25):
    """
    Convert all patch experts for a given scale to CoreML.

    Args:
        model_dir: Base model directory
        scale: Patch scale (0.25, 0.35, or 0.5)

    Returns:
        Dictionary mapping (view_idx, landmark_idx) -> CoreMLPatchExpert
    """
    model_path = Path(model_dir) / f'exported_ccnf_{scale}'

    if not model_path.exists():
        print(f"Model path not found: {model_path}")
        return {}

    coreml_experts = {}

    # Get view directories
    view_dirs = sorted(model_path.glob('view_*'))

    for view_dir in view_dirs:
        view_idx = int(view_dir.name.split('_')[1])

        # Get patch directories
        patch_dirs = sorted(view_dir.glob('patch_*'))

        for patch_dir in patch_dirs:
            landmark_idx = int(patch_dir.name.split('_')[1])

            # Check if patch exists
            if (patch_dir / 'metadata.npz').exists():
                try:
                    expert = CoreMLPatchExpert(str(patch_dir))
                    if expert.coreml_model is not None:
                        coreml_experts[(view_idx, landmark_idx)] = expert
                except Exception as e:
                    print(f"Failed to convert patch {landmark_idx} view {view_idx}: {e}")

    print(f"Converted {len(coreml_experts)} patch experts to CoreML")
    return coreml_experts


def benchmark_coreml():
    """Benchmark CoreML vs Numba patch expert inference."""
    import sys
    sys.path.insert(0, 'pyclnf')

    from pyclnf.core.patch_expert import CCNFPatchExpert

    patch_dir = "pyclnf/pyclnf/models/exported_ccnf_0.25/view_00/patch_30"

    # Load both versions
    cpu_expert = CCNFPatchExpert(patch_dir)
    coreml_expert = CoreMLPatchExpert(patch_dir)

    # Generate test patches
    np.random.seed(42)
    test_patches = [
        np.random.randint(0, 256, (cpu_expert.height, cpu_expert.width), dtype=np.uint8)
        for _ in range(100)
    ]

    # Benchmark CPU (Numba)
    t0 = time.perf_counter()
    for patch in test_patches:
        cpu_expert.compute_response(patch)
    cpu_time = time.perf_counter() - t0

    # Benchmark CoreML
    t0 = time.perf_counter()
    for patch in test_patches:
        coreml_expert.compute_response(patch)
    coreml_time = time.perf_counter() - t0

    # Compare accuracy
    max_error = 0.0
    for patch in test_patches[:10]:
        cpu_resp = cpu_expert.compute_response(patch)
        coreml_resp = coreml_expert.compute_response(patch)
        error = abs(cpu_resp - coreml_resp)
        max_error = max(max_error, error)

    print(f"\nBenchmark Results:")
    print(f"  CPU (Numba): {cpu_time*1000:.2f}ms for 100 patches")
    print(f"  CoreML: {coreml_time*1000:.2f}ms for 100 patches")
    print(f"  Speedup: {cpu_time/coreml_time:.1f}x")
    print(f"  Max error: {max_error:.2e}")


if __name__ == '__main__':
    benchmark_coreml()
