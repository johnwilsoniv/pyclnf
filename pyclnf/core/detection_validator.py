"""
Detection Validator for CLNF

Validates landmark detection using a CNN-based classifier.
Ports the C++ OpenFace DetectionValidator to Python.

The validator:
1. Warps the detected face region using piecewise affine warping (PAW)
2. Normalizes the warped image
3. Runs a CNN classifier to get a confidence score
4. Returns True if confidence > validation_boundary (default 0.725)
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import struct


def read_mat_bin(f) -> np.ndarray:
    """Read a matrix in OpenFace binary format."""
    # Read dimensions
    rows = struct.unpack('i', f.read(4))[0]
    cols = struct.unpack('i', f.read(4))[0]
    dtype = struct.unpack('i', f.read(4))[0]

    # OpenCV type constants:
    # CV_8UC1 = 0, CV_8SC1 = 1, CV_16UC1 = 2, CV_16SC1 = 3
    # CV_32SC1 = 4, CV_32FC1 = 5, CV_64FC1 = 6
    dtype_map = {
        0: (np.uint8, 1),      # CV_8UC1
        1: (np.int8, 1),       # CV_8SC1
        4: (np.int32, 4),      # CV_32SC1
        5: (np.float32, 4),    # CV_32FC1
        6: (np.float64, 8),    # CV_64FC1
    }

    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")

    np_dtype, elem_size = dtype_map[dtype]

    # Read data
    data = np.frombuffer(f.read(rows * cols * elem_size), dtype=np_dtype)
    return data.reshape((rows, cols))


class PAW:
    """Piecewise Affine Warp for face alignment."""

    def __init__(self):
        self.destination_landmarks = None  # Reference landmarks
        self.triangulation = None  # Triangle indices
        self.pixel_mask = None  # Mask of valid pixels
        self.triangle_id = None  # Which triangle each pixel belongs to
        self.alpha = None  # Barycentric coordinates
        self.beta = None
        self.map_x = None  # Precomputed warp maps
        self.map_y = None
        self.coefficients = None  # Affine coefficients per triangle

    def read(self, f):
        """Read PAW from binary stream."""
        # Read number of pixels
        self.num_pixels = struct.unpack('i', f.read(4))[0]

        # Read min coordinates (doubles)
        self.min_x = struct.unpack('d', f.read(8))[0]
        self.min_y = struct.unpack('d', f.read(8))[0]

        # Read destination landmarks
        self.destination_landmarks = read_mat_bin(f).astype(np.float32)

        # Read triangulation
        self.triangulation = read_mat_bin(f).astype(np.int32)

        # Read triangle ID map
        self.triangle_id = read_mat_bin(f).astype(np.int32)

        # Read pixel mask
        pixel_mask_raw = read_mat_bin(f)
        self.pixel_mask = pixel_mask_raw.astype(np.uint8)

        # Read alpha/beta barycentric coordinates
        self.alpha = read_mat_bin(f).astype(np.float32)
        self.beta = read_mat_bin(f).astype(np.float32)

        # Initialize maps
        h, w = self.pixel_mask.shape
        self.map_x = np.zeros((h, w), dtype=np.float32)
        self.map_y = np.zeros((h, w), dtype=np.float32)
        num_triangles = self.triangulation.shape[0]
        self.coefficients = np.zeros((num_triangles, 6), dtype=np.float32)

    def number_of_landmarks(self) -> int:
        return self.destination_landmarks.shape[0] // 2

    def number_of_triangles(self) -> int:
        return self.triangulation.shape[0]

    def calc_coeff(self, source_landmarks: np.ndarray):
        """Calculate affine coefficients for each triangle.

        The coefficients define an affine transform from destination (template)
        coordinates to source (image) coordinates for each triangle.

        For each triangle l with vertices (i, j, k):
          c1, c4 = x, y of vertex i in source image
          c2, c5 = delta x, y from vertex i to j
          c3, c6 = delta x, y from vertex i to k

        The alpha/beta arrays (per triangle, 3 values each) encode the
        barycentric mapping for the affine transform.
        """
        p = self.number_of_landmarks()

        for l in range(self.number_of_triangles()):
            i = self.triangulation[l, 0]
            j = self.triangulation[l, 1]
            k = self.triangulation[l, 2]

            # Get source landmark coordinates
            # Format: source_landmarks is (2*p, 1) - first p are x, second p are y
            if source_landmarks.ndim == 1:
                c1 = source_landmarks[i]        # x of vertex i
                c2 = source_landmarks[j] - c1   # x delta i->j
                c3 = source_landmarks[k] - c1   # x delta i->k
                c4 = source_landmarks[i + p]    # y of vertex i
                c5 = source_landmarks[j + p] - c4  # y delta i->j
                c6 = source_landmarks[k + p] - c4  # y delta i->k
            else:
                c1 = source_landmarks[i, 0]
                c2 = source_landmarks[j, 0] - c1
                c3 = source_landmarks[k, 0] - c1
                c4 = source_landmarks[i + p, 0]
                c5 = source_landmarks[j + p, 0] - c4
                c6 = source_landmarks[k + p, 0] - c4

            # Get barycentric coefficients for this triangle
            c_alpha = self.alpha[l]  # Shape: (3,) - alpha[0], alpha[1], alpha[2]
            c_beta = self.beta[l]    # Shape: (3,) - beta[0], beta[1], beta[2]

            # Compute affine coefficients
            # These map destination coords (xi, yi) to source coords (xo, yo):
            # xo = coeff[0] + coeff[1] * xi + coeff[2] * yi
            # yo = coeff[3] + coeff[4] * xi + coeff[5] * yi
            self.coefficients[l, 0] = c1 + c2 * c_alpha[0] + c3 * c_beta[0]
            self.coefficients[l, 1] = c2 * c_alpha[1] + c3 * c_beta[1]
            self.coefficients[l, 2] = c2 * c_alpha[2] + c3 * c_beta[2]
            self.coefficients[l, 3] = c4 + c5 * c_alpha[0] + c6 * c_beta[0]
            self.coefficients[l, 4] = c5 * c_alpha[1] + c6 * c_beta[1]
            self.coefficients[l, 5] = c5 * c_alpha[2] + c6 * c_beta[2]

    def warp_region(self):
        """Compute warp maps from coefficients.

        Maps each destination pixel (in template space) to its corresponding
        source pixel (in image space) using the precomputed affine coefficients.
        """
        h, w = self.pixel_mask.shape

        for y in range(h):
            # Destination y coordinate (in template space)
            yi = float(y) + self.min_y

            for x in range(w):
                if self.pixel_mask[y, x] == 0:
                    # Outside mask - mark as invalid
                    self.map_x[y, x] = -1
                    self.map_y[y, x] = -1
                else:
                    # Destination x coordinate (in template space)
                    xi = float(x) + self.min_x

                    # Get triangle for this pixel
                    tri_id = self.triangle_id[y, x]

                    if tri_id >= 0 and tri_id < len(self.coefficients):
                        # Get affine coefficients for this triangle
                        coeff = self.coefficients[tri_id]

                        # Apply affine transform to get source coordinates
                        # xo = coeff[0] + coeff[1] * xi + coeff[2] * yi
                        # yo = coeff[3] + coeff[4] * xi + coeff[5] * yi
                        self.map_x[y, x] = coeff[0] + coeff[1] * xi + coeff[2] * yi
                        self.map_y[y, x] = coeff[3] + coeff[4] * xi + coeff[5] * yi
                    else:
                        self.map_x[y, x] = -1
                        self.map_y[y, x] = -1

    def warp(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Warp image using current landmarks."""
        # Reshape landmarks to match expected format (2n x 1)
        if landmarks.ndim == 2 and landmarks.shape[1] == 2:
            # Convert from (68, 2) to (136, 1)
            source_lm = np.concatenate([landmarks[:, 0], landmarks[:, 1]]).reshape(-1, 1)
        else:
            source_lm = landmarks

        # Calculate coefficients
        self.calc_coeff(source_lm)

        # Compute warp maps
        self.warp_region()

        # Warp image
        h, w = self.pixel_mask.shape
        warped = cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR)

        return warped


class DetectionValidator:
    """CNN-based detection validator."""

    def __init__(self, model_path: Optional[str] = None, validation_boundary: float = 0.725):
        self.validation_boundary = validation_boundary
        self.orientations = []
        self.paws = []
        self.mean_images = []
        self.standard_deviations = []
        self.cnn_layer_types = []
        self.cnn_convolutional_layers_weights = []
        self.cnn_fully_connected_layers_weights = []
        self.cnn_fully_connected_layers_biases = []

        if model_path:
            self.read(model_path)

    def read(self, model_path: str):
        """Read validator model from file."""
        with open(model_path, 'rb') as f:
            # Read validator type
            validator_type = struct.unpack('i', f.read(4))[0]
            if validator_type != 3:
                raise ValueError(f"Unsupported validator type: {validator_type}")

            # Read number of views
            n = struct.unpack('i', f.read(4))[0]

            # Read orientations
            self.orientations = []
            for i in range(n):
                orientation = read_mat_bin(f)
                # Convert to radians
                self.orientations.append(orientation.flatten() * np.pi / 180.0)

            # Initialize storage
            self.paws = [None] * n
            self.mean_images = [None] * n
            self.standard_deviations = [None] * n
            self.cnn_layer_types = [[] for _ in range(n)]
            self.cnn_convolutional_layers_weights = [[] for _ in range(n)]
            self.cnn_fully_connected_layers_weights = [[] for _ in range(n)]
            self.cnn_fully_connected_layers_biases = [[] for _ in range(n)]

            # Read validators for each view
            for i in range(n):
                # Read mean image
                mean_img = read_mat_bin(f).astype(np.float32).T
                self.mean_images[i] = mean_img

                # Read standard deviation
                std_dev = read_mat_bin(f).astype(np.float32).T
                self.standard_deviations[i] = std_dev

                # Read network depth
                network_depth = struct.unpack('i', f.read(4))[0]

                for layer in range(network_depth):
                    layer_type = struct.unpack('i', f.read(4))[0]
                    self.cnn_layer_types[i].append(layer_type)

                    if layer_type == 0:  # Convolutional
                        num_in_maps = struct.unpack('i', f.read(4))[0]
                        num_kernels = struct.unpack('i', f.read(4))[0]

                        # Read biases
                        biases = []
                        for k in range(num_kernels):
                            bias = struct.unpack('f', f.read(4))[0]
                            biases.append(bias)

                        # Read kernels
                        kernels = []
                        for in_map in range(num_in_maps):
                            kernels_for_input = []
                            for k in range(num_kernels):
                                kernel = read_mat_bin(f).astype(np.float32)
                                kernels_for_input.append(kernel)
                            kernels.append(kernels_for_input)

                        # Rearrange kernels into weight matrix
                        kernel_h, kernel_w = kernels[0][0].shape
                        weight_matrix = np.zeros((num_in_maps * kernel_h * kernel_w, num_kernels), dtype=np.float32)

                        for k in range(num_kernels):
                            for in_map in range(num_in_maps):
                                k_flat = kernels[in_map][k].T.reshape(-1)
                                start_idx = in_map * kernel_h * kernel_w
                                weight_matrix[start_idx:start_idx + kernel_h * kernel_w, k] = k_flat

                        # Transpose and add bias
                        weight_matrix = weight_matrix.T
                        W = np.ones((weight_matrix.shape[0], weight_matrix.shape[1] + 1), dtype=np.float32)
                        W[:, -1] = biases
                        W[:, :-1] = weight_matrix

                        self.cnn_convolutional_layers_weights[i].append(W.T)

                    elif layer_type == 2:  # Fully connected
                        biases = read_mat_bin(f).astype(np.float32)
                        weights = read_mat_bin(f).astype(np.float32)
                        self.cnn_fully_connected_layers_biases[i].append(biases)
                        self.cnn_fully_connected_layers_weights[i].append(weights)

                # Read PAW
                paw = PAW()
                paw.read(f)
                self.paws[i] = paw

        print(f"✓ Detection validator loaded ({n} views)")

    def get_view_id(self, orientation: np.ndarray) -> int:
        """Get the best view ID for the given orientation."""
        min_dist = float('inf')
        best_view = 0

        for i, view_orient in enumerate(self.orientations):
            dist = np.sum((orientation - view_orient) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_view = i

        return best_view

    def normalise_warped_to_vector(self, warped: np.ndarray, view_id: int) -> np.ndarray:
        """Normalise warped image to feature vector.

        Matches C++ two-stage normalization:
        1. Local normalization: zero-mean, unit-variance based on current image
        2. Global normalization: apply pre-computed mean/std images
        """
        mean_img = self.mean_images[view_id]
        std_dev = self.standard_deviations[view_id]
        mask = self.paws[view_id].pixel_mask

        # C++ iterates over transposed image: for i in cols, for j in rows
        warped_t = warped.T
        mask_t = mask.T

        h, w = warped.shape  # Original dimensions

        # Extract pixels within mask in transposed iteration order
        vec = []
        for i in range(w):  # cols in original = rows in transposed
            for j in range(h):  # rows in original = cols in transposed
                if mask_t[i, j]:
                    vec.append(warped_t[i, j])

        vec = np.array(vec, dtype=np.float32)

        # Step 1: Local normalization (zero-mean, unit-variance)
        local_mean = np.mean(vec)
        local_std = np.std(vec)
        if local_std == 0:
            local_std = 1.0

        vec = (vec - local_mean) / local_std

        # Step 2: Global normalization using pre-computed mean/std
        feature_vec = np.zeros_like(vec)
        for idx in range(len(vec)):
            if idx < mean_img.size:
                feature_vec[idx] = (vec[idx] - mean_img.flat[idx]) / (std_dev.flat[idx] + 1e-6)
            else:
                feature_vec[idx] = vec[idx]

        return feature_vec

    def convolution_direct(self, input_maps: List[np.ndarray], weights: np.ndarray,
                           kernel_h: int, kernel_w: int) -> List[np.ndarray]:
        """Direct convolution using im2col.

        Args:
            input_maps: List of input feature maps
            weights: Weight matrix (kernel^2 * num_in_maps + 1, num_outputs)
                     The +1 row is for bias terms
            kernel_h, kernel_w: Kernel dimensions
        """
        outputs = []

        # Get dimensions
        h, w = input_maps[0].shape
        out_h = h - kernel_h + 1
        out_w = w - kernel_w + 1
        num_outputs = weights.shape[1]  # Number of output feature maps

        # im2col for all input maps
        im2col_data = []
        for inp in input_maps:
            patches = []
            for y in range(out_h):
                for x in range(out_w):
                    patch = inp[y:y+kernel_h, x:x+kernel_w].T.flatten()
                    patches.append(patch)
            im2col_data.append(np.array(patches))

        # Concatenate all input channels
        im2col = np.hstack(im2col_data)
        # Add bias column (multiplies with bias row in weights)
        im2col = np.hstack([im2col, np.ones((im2col.shape[0], 1), dtype=np.float32)])

        # Matrix multiply: (num_patches, kernel^2 * num_in + 1) @ (kernel^2 * num_in + 1, num_out)
        result = im2col @ weights

        # Reshape to output maps
        for k in range(num_outputs):
            out_map = result[:, k].reshape(out_h, out_w)
            outputs.append(out_map)

        return outputs

    def max_pooling(self, input_maps: List[np.ndarray], pool_h: int = 2, pool_w: int = 2,
                    stride_h: int = 2, stride_w: int = 2) -> List[np.ndarray]:
        """Max pooling layer."""
        outputs = []

        for inp in input_maps:
            h, w = inp.shape
            out_h = (h - pool_h) // stride_h + 1
            out_w = (w - pool_w) // stride_w + 1

            out = np.zeros((out_h, out_w), dtype=np.float32)
            for y in range(out_h):
                for x in range(out_w):
                    region = inp[y*stride_h:y*stride_h+pool_h, x*stride_w:x*stride_w+pool_w]
                    out[y, x] = np.max(region)

            outputs.append(out)

        return outputs

    def fully_connected(self, input_maps: List[np.ndarray], weights: np.ndarray,
                        biases: np.ndarray) -> List[np.ndarray]:
        """Fully connected layer."""
        # Flatten all inputs - C++ transposes before flatten (column-major order)
        # So we use 'F' order (Fortran/column-major) to match
        flat_input = np.concatenate([inp.flatten('F') for inp in input_maps])

        # Compute output
        output = weights.T @ flat_input + biases.flatten()

        return [output.reshape(1, -1)]

    def check_cnn(self, warped: np.ndarray, view_id: int) -> float:
        """Run CNN on warped image and return score."""
        # Normalise to vector
        feature_vec = self.normalise_warped_to_vector(warped, view_id)

        # Reconstruct image from vector (using transposed iteration order like C++)
        mask = self.paws[view_id].pixel_mask
        h, w = mask.shape
        img = np.zeros((h, w), dtype=np.float32)
        img_t = img.T
        mask_t = mask.T

        idx = 0
        for i in range(w):  # cols
            for j in range(h):  # rows
                if mask_t[i, j] and idx < len(feature_vec):
                    img_t[i, j] = feature_vec[idx]
                    idx += 1
        img = img_t.T

        # Run CNN
        input_maps = [img]
        cnn_layer = 0
        fc_layer = 0

        for layer_type in self.cnn_layer_types[view_id]:
            if layer_type == 0:  # Convolution
                weights = self.cnn_convolutional_layers_weights[view_id][cnn_layer]
                # Infer kernel size from weights
                # weights shape: (kernel_h * kernel_w * num_in_maps + 1, num_outputs)
                num_in_maps = len(input_maps)
                kernel_size_sq = (weights.shape[0] - 1) // num_in_maps  # -1 for bias row
                kernel_h = kernel_w = int(np.sqrt(kernel_size_sq))
                input_maps = self.convolution_direct(input_maps, weights, kernel_h, kernel_w)
                cnn_layer += 1

            elif layer_type == 1:  # Max pooling
                input_maps = self.max_pooling(input_maps)

            elif layer_type == 2:  # Fully connected
                weights = self.cnn_fully_connected_layers_weights[view_id][fc_layer]
                biases = self.cnn_fully_connected_layers_biases[view_id][fc_layer]
                input_maps = self.fully_connected(input_maps, weights, biases)
                fc_layer += 1

            elif layer_type == 3:  # ReLU
                input_maps = [np.maximum(0, inp) for inp in input_maps]

            elif layer_type == 4:  # Sigmoid
                input_maps = [1.0 / (1.0 + np.exp(-inp)) for inp in input_maps]

        # Convert quantized output to continuous value
        # The output is a probability distribution over bins
        output = input_maps[0].flatten()
        max_idx = np.argmax(output)
        bins = len(output)
        max_val = 1.0
        min_val = -1.0
        step_size = (max_val - min_val) / bins
        unquantized = min_val + step_size / 2.0 + max_idx * step_size

        # Convert to confidence (0 = low, 1 = high)
        confidence = 0.5 * (1.0 - unquantized)

        return confidence

    def check(self, orientation: np.ndarray, image: np.ndarray,
              landmarks: np.ndarray) -> float:
        """
        Check if detected landmarks are valid.

        Args:
            orientation: Head pose [pitch, yaw, roll] in radians
            image: Grayscale image
            landmarks: Detected landmarks (68, 2) or (136,)

        Returns:
            Confidence score (0-1), where > validation_boundary indicates success
        """
        view_id = self.get_view_id(orientation)
        paw = self.paws[view_id]

        # Convert landmarks to (136, 1) format if needed
        if landmarks.ndim == 2 and landmarks.shape[1] == 2:
            source_lm = np.concatenate([landmarks[:, 0], landmarks[:, 1]]).reshape(-1, 1).astype(np.float32)
        else:
            source_lm = landmarks.astype(np.float32).reshape(-1, 1)

        n_pts = len(source_lm) // 2
        xs = source_lm[:n_pts, 0]
        ys = source_lm[n_pts:, 0]

        # Extract bounding box (matching C++)
        min_x_f, max_x_f = xs.min(), xs.max()
        min_y_f, max_y_f = ys.min(), ys.max()

        # Add padding for bilinear interpolation
        min_x = int(min_x_f - 3.0)
        max_x = int(max_x_f + 3.0)
        min_y = int(min_y_f - 3.0)
        max_y = int(max_y_f + 3.0)

        # Clamp to image bounds
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(image.shape[1] - 1, max_x)
        max_y = min(image.shape[0] - 1, max_y)

        # Check for valid ROI
        if max_x - min_x <= 1 or max_y - min_y <= 1:
            return 0.0

        # Crop image to ROI and convert to float32 (matching C++)
        roi = image[min_y:max_y, min_x:max_x].astype(np.float32)

        # Adjust landmarks to ROI coordinates
        adjusted_lm = source_lm.copy()
        adjusted_lm[:n_pts, 0] -= min_x
        adjusted_lm[n_pts:, 0] -= min_y

        # Warp face region
        warped = paw.warp(roi, adjusted_lm)

        # Run CNN
        confidence = self.check_cnn(warped, view_id)

        return confidence

    def validate(self, orientation: np.ndarray, image: np.ndarray,
                 landmarks: np.ndarray) -> bool:
        """
        Validate landmarks.

        Returns:
            True if landmarks are valid, False otherwise
        """
        confidence = self.check(orientation, image, landmarks)
        return confidence > self.validation_boundary
