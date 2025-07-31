# Vulkan Triangle Example

A relatively simple, well-commented Vulkan example that draws a triangle. Perfect for learning the Vulkan API step-by-step.

## Features

- **Single C++ file**: All Vulkan code is in `src/main.cpp` with detailed comments explaining each part of the process.
- **GLSL Shaders**: Vertex and fragment shaders are written in GLSL and compiled to SPIR-V using the Makefile.
- **Cross-platform**: Builds and runs on both Windows and Linux.
- **Minimal Dependencies**: Uses GLFW for window management. A prebuilt `glfw3` is included for Windows.

## Prerequisites

### Windows

- Install the [Vulkan SDK](https://vulkan.lunarg.com/) from LunarG.
- Ensure your GPU drivers support Vulkan and are up to date.
- A C++ compiler supporting C++17 or newer (e.g., MSVC from Visual Studio or MinGW).

### Linux

- Install the Vulkan SDK and GLFW development libraries:
```bash
    sudo apt install vulkan-tools libvulkan-dev vulkan-validationlayers-dev spirv-tools libglfw3-dev libglm-dev
```
- On Arch Linux / Manjaro, install Vulkan and GLFW packages:
```bash
    sudo pacman -S glfw vulkan-devel glm
```
- Make sure your GPU drivers support Vulkan (Mesa for AMD/Intel or proprietary drivers for NVIDIA).
- A C++ compiler with C++17 support (e.g., g++ or clang++).

## Building the Project

1. **Clone the repository:**
```bash
    git clone https://your-repo-url.git  
    cd your-repo-directory
```
2. **Build with CMake:**
```bash
    mkdir build
    cd build
    cmake
    make
```
3. **Run the executable:**
```basg
    ./VulkanTriangle
```
The cmake compiles GLSL shaders using `glslangValidator` (make sure it’s in your PATH, usually provided by the Vulkan SDK) and copies the compiled SPIR-V files next to the executable.

## Code Structure and Highlights

- **`main.cpp`**: Contains all Vulkan initialization, including instance creation, physical and logical device selection, swapchain setup, pipeline creation, command buffer recording, and synchronization.
- **`vertex_shader.vert`**: Basic vertex shader positioning triangle vertices. Also generates the rainbow colors using the rasterizer.
- **`fragment_shader.frag`**: Fragment shader setting the triangle color.

## Troubleshooting

- **Validation layers**: Run with validation layers enabled to catch Vulkan API misuse or errors. The example enables these layers in debug mode.
- **Driver issues**: Make sure your GPU drivers are up to date and support Vulkan.
- **Debug output**: The application prints Vulkan debug messages to the console when validation layers are enabled.

## License
The glfw3 code uses the zlib License.
