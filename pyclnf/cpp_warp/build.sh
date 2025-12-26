#!/bin/bash
# Build script for cpp_warp extension module

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building cpp_warp extension..."
echo "================================"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. \
    -DPython3_EXECUTABLE=/Library/Frameworks/Python.framework/Versions/3.10/bin/python3 \
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release -j$(sysctl -n hw.ncpu)

# Copy the built module to the cpp_warp directory
cp cpp_warp*.so ../ 2>/dev/null || true

echo ""
echo "Build complete!"
echo "================================"

# Verify the build
cd ..
echo "Testing import..."
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3 -c "
import cpp_warp
print(f'cpp_warp loaded successfully!')
print(f'OpenCV version: {cpp_warp.get_opencv_version()}')
"
