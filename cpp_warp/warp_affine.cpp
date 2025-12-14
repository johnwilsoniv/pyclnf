/**
 * C++ warpAffine wrapper for pyclnf
 *
 * This module provides a Python-callable warpAffine function that uses
 * the exact same OpenCV library as C++ OpenFace (Homebrew OpenCV 4.12).
 *
 * This eliminates numerical differences caused by different OpenCV builds
 * between Python's opencv-python package and Homebrew's OpenCV.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

namespace py = pybind11;

/**
 * Perform warpAffine using Homebrew's OpenCV 4.12
 *
 * This matches exactly what C++ OpenFace uses for patch extraction.
 *
 * @param src Source image (float32, grayscale)
 * @param M 2x3 transformation matrix (float32 or float64)
 * @param dsize Output size (width, height)
 * @param flags Interpolation flags (default: WARP_INVERSE_MAP | INTER_LINEAR)
 * @param borderMode Border handling mode (default: BORDER_CONSTANT)
 * @param borderValue Border fill value (default: 0)
 * @return Warped image as numpy array
 */
// Default flags: WARP_INVERSE_MAP (16) | INTER_LINEAR (1) = 17
// Default borderMode: BORDER_CONSTANT = 0
constexpr int DEFAULT_FLAGS = 17;  // cv::WARP_INVERSE_MAP | cv::INTER_LINEAR
constexpr int DEFAULT_BORDER_MODE = 0;  // cv::BORDER_CONSTANT

py::array_t<float> warp_affine(
    py::array_t<float, py::array::c_style | py::array::forcecast> src,
    py::array_t<double, py::array::c_style | py::array::forcecast> M,
    std::tuple<int, int> dsize,
    int flags = DEFAULT_FLAGS,
    int borderMode = DEFAULT_BORDER_MODE,
    double borderValue = 0.0
) {
    // Get input array info
    py::buffer_info src_buf = src.request();
    py::buffer_info M_buf = M.request();

    // Validate dimensions
    if (src_buf.ndim != 2) {
        throw std::runtime_error("Source image must be 2D (grayscale)");
    }
    if (M_buf.ndim != 2 || M_buf.shape[0] != 2 || M_buf.shape[1] != 3) {
        throw std::runtime_error("Transformation matrix must be 2x3");
    }

    // Create cv::Mat wrappers (no copy)
    cv::Mat src_mat(
        static_cast<int>(src_buf.shape[0]),
        static_cast<int>(src_buf.shape[1]),
        CV_32FC1,
        src_buf.ptr
    );

    cv::Mat M_mat(
        2, 3,
        CV_64FC1,
        M_buf.ptr
    );

    // Output size
    int out_width = std::get<0>(dsize);
    int out_height = std::get<1>(dsize);

    // Perform warpAffine
    cv::Mat dst;
    cv::warpAffine(
        src_mat,
        dst,
        M_mat,
        cv::Size(out_width, out_height),
        flags,
        borderMode,
        cv::Scalar(borderValue)
    );

    // Create output numpy array
    py::array_t<float> result({dst.rows, dst.cols});
    py::buffer_info result_buf = result.request();

    // Copy result (dst may not be contiguous)
    std::memcpy(result_buf.ptr, dst.data, dst.rows * dst.cols * sizeof(float));

    return result;
}

/**
 * Extract Area of Interest (AOI) patch around a landmark
 *
 * This is the exact operation used in CLNF patch extraction.
 * Matches C++ OpenFace's patch extraction exactly.
 *
 * @param image Source grayscale image (float32)
 * @param center_x Landmark X coordinate in image space
 * @param center_y Landmark Y coordinate in image space
 * @param sim_ref_to_img 2x3 similarity transform from reference to image
 * @param aoi_size Size of the AOI patch (square)
 * @return Extracted AOI patch
 */
py::array_t<float> extract_aoi(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    double center_x,
    double center_y,
    py::array_t<double, py::array::c_style | py::array::forcecast> sim_ref_to_img,
    int aoi_size
) {
    py::buffer_info img_buf = image.request();
    py::buffer_info sim_buf = sim_ref_to_img.request();

    if (img_buf.ndim != 2) {
        throw std::runtime_error("Image must be 2D (grayscale)");
    }
    if (sim_buf.ndim != 2 || sim_buf.shape[0] != 2 || sim_buf.shape[1] != 3) {
        throw std::runtime_error("Similarity transform must be 2x3");
    }

    // Get transform components
    double* sim_ptr = static_cast<double*>(sim_buf.ptr);
    double a1 = sim_ptr[0];   // sim_ref_to_img[0, 0]
    double b1 = -sim_ptr[1];  // -sim_ref_to_img[0, 1]

    // Compute AOI transform (matches C++ OpenFace exactly)
    double center_offset = (aoi_size - 1.0) / 2.0;
    double tx = center_x - a1 * center_offset + b1 * center_offset;
    double ty = center_y - a1 * center_offset - b1 * center_offset;

    // Build 2x3 transform matrix
    cv::Mat M = (cv::Mat_<double>(2, 3) <<
        a1, -b1, tx,
        b1,  a1, ty
    );

    // Create cv::Mat wrapper for source image
    cv::Mat src_mat(
        static_cast<int>(img_buf.shape[0]),
        static_cast<int>(img_buf.shape[1]),
        CV_32FC1,
        img_buf.ptr
    );

    // Perform warpAffine with WARP_INVERSE_MAP (matches C++ OpenFace)
    cv::Mat dst;
    cv::warpAffine(
        src_mat,
        dst,
        M,
        cv::Size(aoi_size, aoi_size),
        cv::WARP_INVERSE_MAP | cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );

    // Create output numpy array
    py::array_t<float> result({aoi_size, aoi_size});
    py::buffer_info result_buf = result.request();
    std::memcpy(result_buf.ptr, dst.data, aoi_size * aoi_size * sizeof(float));

    return result;
}

/**
 * Get OpenCV version string
 */
std::string get_opencv_version() {
    return cv::getVersionString();
}

/**
 * Get OpenCV build information
 */
std::string get_opencv_build_info() {
    return cv::getBuildInformation();
}

// Module definition
PYBIND11_MODULE(cpp_warp, m) {
    m.doc() = R"pbdoc(
        C++ warpAffine wrapper for pyclnf

        This module provides warpAffine operations using Homebrew's OpenCV 4.12,
        ensuring exact numerical matching with C++ OpenFace.
    )pbdoc";

    m.def("warp_affine", &warp_affine,
          py::arg("src"),
          py::arg("M"),
          py::arg("dsize"),
          py::arg("flags") = DEFAULT_FLAGS,
          py::arg("borderMode") = DEFAULT_BORDER_MODE,
          py::arg("borderValue") = 0.0,
          R"pbdoc(
              Perform warpAffine using Homebrew's OpenCV 4.12.

              Parameters:
                  src: Source image (float32, grayscale)
                  M: 2x3 transformation matrix
                  dsize: Output size (width, height)
                  flags: Interpolation flags (default: WARP_INVERSE_MAP | INTER_LINEAR)
                  borderMode: Border handling (default: BORDER_CONSTANT)
                  borderValue: Border fill value (default: 0)

              Returns:
                  Warped image as numpy array
          )pbdoc");

    m.def("extract_aoi", &extract_aoi,
          py::arg("image"),
          py::arg("center_x"),
          py::arg("center_y"),
          py::arg("sim_ref_to_img"),
          py::arg("aoi_size"),
          R"pbdoc(
              Extract Area of Interest patch around a landmark.

              This matches C++ OpenFace's patch extraction exactly.

              Parameters:
                  image: Source grayscale image (float32)
                  center_x: Landmark X coordinate
                  center_y: Landmark Y coordinate
                  sim_ref_to_img: 2x3 similarity transform
                  aoi_size: Size of the AOI patch

              Returns:
                  Extracted AOI patch
          )pbdoc");

    m.def("get_opencv_version", &get_opencv_version,
          "Get the OpenCV version this module was built against");

    m.def("get_opencv_build_info", &get_opencv_build_info,
          "Get OpenCV build information");

    // Export OpenCV constants for convenience (as integers)
    m.attr("INTER_LINEAR") = py::int_(1);
    m.attr("INTER_CUBIC") = py::int_(2);
    m.attr("INTER_LANCZOS4") = py::int_(4);
    m.attr("WARP_INVERSE_MAP") = py::int_(16);
    m.attr("BORDER_CONSTANT") = py::int_(0);
    m.attr("BORDER_REPLICATE") = py::int_(1);
    m.attr("BORDER_REFLECT") = py::int_(2);
}
