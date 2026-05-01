cmake_minimum_required(VERSION 3.15)
project(cpp_warp LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Find pybind11
# NOTE: pybind11 3.x's `python -m pybind11 --cmakedir` returns the path wrapped
# in literal Python repr quotes (e.g. "'/path/to/cmake'"), which breaks CMake's
# OUTPUT_VARIABLE capture. Use `pybind11.get_cmake_dir()` instead — clean string.
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

# Find OpenCV from Homebrew
# This is the key - we want the EXACT same OpenCV that C++ OpenFace uses
set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# Create the Python module
pybind11_add_module(cpp_warp warp_affine.cpp)

# Link against OpenCV
target_link_libraries(cpp_warp PRIVATE ${OpenCV_LIBS})

# Include OpenCV headers
target_include_directories(cpp_warp PRIVATE ${OpenCV_INCLUDE_DIRS})

# Set output directory
set_target_properties(cpp_warp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# On macOS, set the rpath to find OpenCV libraries at runtime
if(APPLE)
    set_target_properties(cpp_warp PROPERTIES
        INSTALL_RPATH "@loader_path;/opt/homebrew/opt/opencv/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()
