cmake_minimum_required(VERSION 3.10)
project(VulkanTriangle)

set(CMAKE_CXX_STANDARD 17)

# === FIND VULKAN ===
find_package(Vulkan QUIET)

# === FIND GLFW3 via system-wide install (Linux, or vcpkg on Windows) ===
find_package(glfw3 CONFIG QUIET)

# === MANUAL GLFW3 SETUP (Windows only fallback) ===
if(NOT glfw3_FOUND AND WIN32)
    message(STATUS "GLFW3 not found via package manager. Trying local fallback...")

    set(GLFW_LOCAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/glfw3")

    find_path(GLFW_INCLUDE_DIR
        NAMES GLFW/glfw3.h
        PATHS ${GLFW_LOCAL_PATH}/include
        NO_DEFAULT_PATH
    )

    find_library(GLFW_LIBRARY
        NAMES glfw3 glfw3dll glfw
        PATHS
            ${GLFW_LOCAL_PATH}/lib-vc2022
            ${GLFW_LOCAL_PATH}/lib-vc2019
            ${GLFW_LOCAL_PATH}/lib
        NO_DEFAULT_PATH
    )

    if(GLFW_INCLUDE_DIR AND GLFW_LIBRARY)
        message(STATUS "Using local GLFW3 include: ${GLFW_INCLUDE_DIR}")
        message(STATUS "Using local GLFW3 library: ${GLFW_LIBRARY}")
        add_library(glfw_custom INTERFACE)
        target_include_directories(glfw_custom INTERFACE ${GLFW_INCLUDE_DIR})
        target_link_libraries(glfw_custom INTERFACE ${GLFW_LIBRARY})
        set(glfw3_FOUND TRUE) # Mark as found so we can use consistent logic
    else()
        message(FATAL_ERROR "Local GLFW 3.4 not found or incomplete (missing include/lib).")
    endif()
endif()

# === MANUAL VULKAN SETUP ===
if(NOT Vulkan_FOUND)
    message(STATUS "Vulkan not found via find_package. Checking manual setup...")

    if(DEFINED ENV{VULKAN_SDK})
        set(VULKAN_SDK_PATH $ENV{VULKAN_SDK})
    endif()

    if(VULKAN_SDK_PATH)
        include_directories(${VULKAN_SDK_PATH}/Include)
        link_directories(${VULKAN_SDK_PATH}/Lib)
        set(VULKAN_LIBRARY vulkan-1)
        set(VULKAN_FOUND TRUE)
        message(STATUS "Using Vulkan SDK at: ${VULKAN_SDK_PATH}")
    endif()
endif()

# === FAIL EARLY IF NOT FOUND ===
if(NOT glfw3_FOUND)
    message(FATAL_ERROR "GLFW3 not found. Please:\n"
        "1. Install system-wide (e.g., libglfw3-dev), or\n"
        "2. Use vcpkg, or\n"
        "3. Place GLFW 3.4 in 'glfw-3.4' folder with include/ and lib/")
endif()

if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan not found. Please install Vulkan SDK and set VULKAN_SDK environment variable.")
endif()

# === EXECUTABLE AND SHADERS ===
add_executable(${PROJECT_NAME}
    main.cpp
    vertex_shader.vert
    fragment_shader.frag
)

# === LINK VULKAN ===
if(TARGET Vulkan::Vulkan)
    target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE ${VULKAN_LIBRARY})
endif()

# === LINK GLFW ===
if(TARGET glfw)
    target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
elseif(TARGET glfw_custom)
    target_link_libraries(${PROJECT_NAME} PRIVATE glfw_custom)
else()
    message(FATAL_ERROR "No GLFW target (glfw or glfw_custom) found. Check configuration.")
endif()

# === PLATFORM-SPECIFIC SETTINGS ===
if(UNIX AND NOT WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE pthread dl xcb X11 X11-xcb)
elseif(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE VK_USE_PLATFORM_WIN32_KHR)
endif()

# === SHADER COMPILATION ===
find_program(GLSLANG_VALIDATOR
    NAMES glslangValidator glslangValidator.exe
    PATHS
        $ENV{VULKAN_SDK}/Bin
        $ENV{VULKAN_SDK}/bin
        ${VULKAN_SDK_PATH}/Bin
        ${VULKAN_SDK_PATH}/bin
        /usr/bin
        /usr/local/bin
)

if(NOT GLSLANG_VALIDATOR)
    message(FATAL_ERROR "glslangValidator not found")
endif()

# Compile vertex shader
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vertex_shader.vert.spv
    COMMAND ${GLSLANG_VALIDATOR} -V ${CMAKE_CURRENT_SOURCE_DIR}/vertex_shader.vert -o ${CMAKE_CURRENT_BINARY_DIR}/vertex_shader.vert.spv
    DEPENDS vertex_shader.vert
    COMMENT "Compiling vertex shader"
)

# Compile fragment shader
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fragment_shader.frag.spv
    COMMAND ${GLSLANG_VALIDATOR} -V ${CMAKE_CURRENT_SOURCE_DIR}/fragment_shader.frag -o ${CMAKE_CURRENT_BINARY_DIR}/fragment_shader.frag.spv
    DEPENDS fragment_shader.frag
    COMMENT "Compiling fragment shader"
)

# Add shader target
add_custom_target(shaders ALL
    DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/vertex_shader.vert.spv
    ${CMAKE_CURRENT_BINARY_DIR}/fragment_shader.frag.spv
)

add_dependencies(${PROJECT_NAME} shaders)

# Copy compiled shaders
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/vertex_shader.vert.spv
    ${CMAKE_CURRENT_BINARY_DIR}/fragment_shader.frag.spv
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)
