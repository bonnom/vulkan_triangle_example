// === VULKAN "HELLO TRIANGLE" APPLICATION ===
//
// INTRODUCTION TO VULKAN:
// Vulkan is a low-level graphics API that eliminates the global context model
// found in OpenGL. Instead of changing rendering state through commands at runtime,
// the programmer must explicitly construct and combine all required pipeline state
// up front. This shifts responsibility from the driver to the application.
// while in OpenGL the driver checks if the current state has been used before,
// and if not, it may need to reconfigure or compile internal state before drawing.
//
// KEY VULKAN CONCEPTS:
// - Explicit Control: No default state - you must configure everything
// - Multi-threading: Designed for CPU parallelization from the ground up
// - Predictable Performance: Minimal driver magic, consistent frame times
// - Cross-platform: One API for Windows, Linux, Android, and other platforms
//
// VULKAN ARCHITECTURE:
// 1. Instance: Application connection to Vulkan runtime
// 2. Physical Device: Available GPU hardware enumeration
// 3. Logical Device: Application's interface to selected GPU
// 4. Queues: Command submission endpoints (graphics, compute, transfer)
// 5. Swapchain: Frame presentation system for window rendering
// 6. Pipelines: Pre-compiled rendering state (shaders, blending, etc.)
// 7. Command Buffers: Recorded GPU commands for batch submission
// 8. Synchronization: Explicit fences, semaphores for CPU/GPU coordination
//
// RENDERING FLOW:
// 1. Record commands in command buffers (once, or every frame)
// 2. Submit command buffers to GPU queues
// 3. GPU executes commands asynchronously
// 4. Present rendered images to screen via swapchain
// 5. Synchronize CPU and GPU using fences/semaphores
//
// WHY VULKAN IS COMPLEX:
// - 1000+ lines for a triangle vs ~10 lines in OpenGL
// - Explicit memory management
// - No error recovery - invalid API usage crashes
// - Platform-specific window system integration
// - Pre-compiled shaders (SPIR-V bytecode)
//
// BENEFITS:
// - Maximum performance and control
// - Multi-threaded command recording
// - Consistent performance across vendors
// - Modern GPU features exposed directly
//
// This example demonstrates the complete initialization sequence needed
// to render a single colored triangle using Vulkan's explicit API.

// === MAIN APPLICATION SOURCE FILE ===
// This is the complete implementation of a Vulkan "Hello Triangle" application.
// It demonstrates the full Vulkan initialization pipeline and rendering loop.
//
// FILE STRUCTURE OVERVIEW:
// - Header includes for Vulkan, GLFW, and standard C++ libraries
// - Configuration constants for window size, validation layers, etc.
// - Helper functions for Vulkan extension loading
// - Data structures for Vulkan resource management
// - Main application class implementing the complete Vulkan pipeline
// - Entry point function (main)
//
// WHY THIS STRUCTURE MATTERS:
// Vulkan requires explicit management of every aspect of graphics rendering.
// Unlike simpler APIs, there's no default state or automatic resource management.
// Every object must be created, used, and destroyed manually.

#define NOMINMAX

// === VULKAN AND WINDOW SYSTEM INCLUDES ===
// Vulkan API header - provides all Vulkan function declarations and types
#include <vulkan/vulkan.h>

// GLFW header - cross-platform window and input library
// Used for window creation and OS integration
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// GLM headers not actually used in this simple example, can be removed
//#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>

// === STANDARD C++ LIBRARIES ===
#include <iostream>    // For console output (error messages, debug info)
#include <fstream>     // For reading shader files from disk
#include <vector>      // Dynamic arrays for managing collections of Vulkan objects
#include <stdexcept>   // Exception handling for error conditions
#include <array>       // Fixed-size arrays (rarely used in this example)
#include <cstring>     // String operations (strcmp for extension checking)
#include <set>         // Unique collections (used for queue family deduplication)
#include <cstdint>     // Fixed-width integer types (uint32_t, etc.)
#include <algorithm>   // Standard algorithms (std::max, std::min)
#include <optional>    // Optional values (used for queue family indices)

// === APPLICATION CONFIGURATION CONSTANTS ===
// Window dimensions - determines initial swapchain size
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// Frame buffering strategy - how many frames can be "in flight" simultaneously
// 2 = Triple buffering (current frame + one queued + one being displayed)
const int MAX_FRAMES_IN_FLIGHT = 2;

// Validation layers for development - catch API usage errors
// VK_LAYER_KHRONOS_validation is the standard validation layer bundle
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// Required device extensions - VK_KHR_swapchain is essential for window presentation
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME  // Enables swapchain functionality
};

// Conditional compilation - enable validation only in debug builds
#ifdef NDEBUG
const bool enableValidationLayers = false;  // Release build - no validation overhead
#else
const bool enableValidationLayers = true;   // Debug build - full validation enabled
#endif

// === VULKAN EXTENSION HELPER FUNCTIONS ===
// Vulkan extension functions must be loaded manually at runtime
// These helpers wrap the extension loading process for debug messaging

// Creates a debug messenger object for validation layer output
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    // Load the extension function pointer from the Vulkan instance
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        // Extension is available - call the actual function
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        // Extension not supported - return error
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

// Destroys a debug messenger object
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    // Load the extension function pointer
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        // Extension is available - call the actual function
        func(instance, debugMessenger, pAllocator);
    }
    // If extension isn't available, do nothing (silent failure)
}

// === HELPER DATA STRUCTURES ===
// Custom structures to organize Vulkan resource information

// Queue family indices - tracks which queue families support which operations
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;  // Queue family that supports graphics operations
    std::optional<uint32_t> presentFamily;   // Queue family that supports presentation to surface
    
    // Check if both required queue families were found
    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// Swapchain support details - collects all capabilities of a device/surface combination
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;     // Basic surface capabilities
    std::vector<VkSurfaceFormatKHR> formats;   // Supported pixel formats
    std::vector<VkPresentModeKHR> presentModes; // Supported presentation modes
};

// === MAIN APPLICATION CLASS ===
// Encapsulates the entire Vulkan application state and functionality
class HelloTriangleApplication {
public:
    // Public interface - main entry point for the application
    void run() {
        initWindow();      // Step 1: Create window and setup GLFW
        initVulkan();      // Steps 2-13: Complete Vulkan initialization pipeline
        mainLoop();        // Step 14: Enter rendering loop
        cleanup();         // Final step: Clean up all Vulkan resources
    }

private:
    // === WINDOW SYSTEM RESOURCES ===
    GLFWwindow* window;  // Platform window handle

    // === VULKAN CORE RESOURCES ===
    VkInstance instance;                    // Vulkan instance (application connection)
    VkDebugUtilsMessengerEXT debugMessenger; // Debug callback handler
    VkSurfaceKHR surface;                   // Window surface connection

    // === GPU RESOURCES ===
    VkPhysicalDevice physicalDevice;        // Selected GPU hardware
    VkDevice device;                        // Logical device connection to GPU

    // === QUEUE HANDLES ===
    VkQueue graphicsQueue;                  // Queue for graphics commands
    VkQueue presentQueue;                   // Queue for presentation commands

    // === SWAPCHAIN RESOURCES ===
    VkSwapchainKHR swapChain;               // Swapchain for frame presentation
    std::vector<VkImage> swapChainImages;   // Actual swapchain images
    VkFormat swapChainImageFormat;          // Pixel format of swapchain images
    VkExtent2D swapChainExtent;             // Dimensions of swapchain images
    std::vector<VkImageView> swapChainImageViews;     // Views for using images
    std::vector<VkFramebuffer> swapChainFramebuffers; // Framebuffers for rendering

    // === RENDERING PIPELINE RESOURCES ===
    VkRenderPass renderPass;                // Rendering workflow definition
    VkPipelineLayout pipelineLayout;        // Pipeline interface (uniforms, etc.)
    VkPipeline graphicsPipeline;            // Complete graphics rendering pipeline

    // === COMMAND EXECUTION RESOURCES ===
    VkCommandPool commandPool;              // Memory pool for command buffers
    std::vector<VkCommandBuffer> commandBuffers; // Recorded rendering commands

    // === SYNCHRONIZATION PRIMITIVES ===
    std::vector<VkSemaphore> imageAvailableSemaphores;   // Signal when image is ready
    std::vector<VkSemaphore> renderFinishedSemaphores;   // Signal when rendering done
    std::vector<VkFence> inFlightFences;                 // CPU waits for frame completion
    std::vector<VkFence> imagesInFlight;                 // Track which fence owns each image
    size_t currentFrame = 0;                                 // Current frame index for triple buffering

    // === RESIZE HANDLING ===
    bool framebufferResized;  // Flag indicating window resize occurred

    // === WINDOW MANAGEMENT ===
    void initWindow() {
        glfwInit();

        // Tell GLFW not to create an OpenGL context (we're using Vulkan)
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        // Create the actual window with specified dimensions
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        
        // Set up callback for window resize events
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    // Static callback function for window resize events
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        // Get the application instance from GLFW user pointer
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        // Mark that a resize occurred
        app->framebufferResized = true;
    }

    // === MAIN VULKAN SETTING UP PIPELINE ===
    void initVulkan() {
        createInstance();           // Step 1: Create Vulkan instance
        setupDebugMessenger();      // Step 2: Setup debug callbacks
        createSurface();            // Step 3: Create window surface
        pickPhysicalDevice();       // Step 4: Select GPU
        createLogicalDevice();      // Step 5: Create logical device
        createSwapChain();          // Step 6: Setup swapchain
        createImageViews();         // Step 7: Create image views
        createRenderPass();         // Step 8: Define render pass
        createGraphicsPipeline();   // Step 9: Create graphics pipeline
        createFramebuffers();       // Step 10: Create framebuffers
        createCommandPool();        // Step 11: Create command pool
        createCommandBuffers();     // Step 12: Allocate command buffers
        createSyncObjects();        // Step 13: Create synchronization primitives
    }

    // === MAIN RENDER LOOP ===
    void mainLoop() {
        // Continue rendering until window is closed
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();  // Process window events (input, resize, etc.)
            drawFrame();       // Render one frame
        }

        // Wait for all GPU work to complete before cleanup
        vkDeviceWaitIdle(device);
    }

    // === SWAPCHAIN CLEANUP (PARTIAL CLEANUP FOR RESIZE/RECREATION) ===
    // This function cleans up only the Vulkan objects that are directly dependent 
    // on the swapchain and need to be recreated when the window is resized or 
    // the swapchain becomes invalid. This is a "partial cleanup" used during
    // runtime when we need to rebuild the rendering pipeline.
    //
    // ORDER MATTERS: Vulkan object destruction must follow reverse creation order
    // to avoid dangling references. For example, you must destroy framebuffers
    // before destroying the render pass they depend on.
    //
    // DEPENDENCY CHAIN: 
    // Framebuffers → Render Pass → Pipeline → Pipeline Layout → Image Views → Swapchain
    void cleanupSwapChain() {
        // 1. Destroy framebuffers first (they reference the render pass)
        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        // 2. Free command buffers (they may reference the pipeline)
        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

        // 3. Destroy graphics pipeline objects (pipeline depends on render pass)
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        // 4. Destroy image views (they reference swapchain images)
        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        // 5. Finally destroy the swapchain itself
        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    // === COMPLETE APPLICATION CLEANUP (SHUTDOWN) ===
    // This function performs a complete cleanup of ALL Vulkan resources when the
    // application is shutting down. It calls cleanupSwapChain() first to handle
    // the swapchain-dependent objects, then destroys the remaining resources.
    //
    // CLEANUP ORDER: 
    // 1. Swapchain objects (handled by cleanupSwapChain)
    // 2. Synchronization primitives (semaphores, fences)
    // 3. Command pools
    // 4. Logical device
    // 5. Debug messenger (if validation enabled)
    // 6. Surface and instance
    // 7. Window system (GLFW)
    //
    // WHY THIS ORDER MATTERS:
    // - Vulkan objects must be destroyed before objects they depend on
    // - You cannot destroy a device while command buffers are still allocated
    // - Instance must outlive all objects created from it
    void cleanup() {
        // First clean up all swapchain-dependent resources
        cleanupSwapChain();
        // Destroy synchronization objects used for frame timing
        // These are created per-frame to prevent pipeline stalls
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);  // CPU → GPU signaling
            vkDestroyFence(device, inFlightFences[i], nullptr);                // GPU completion tracking
        }
        // Destroy render finished semaphores (one per swapchain image)
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);  // GPU → CPU signaling
        }
        // Destroy command pool (this also frees all command buffers)
        vkDestroyCommandPool(device, commandPool, nullptr);
        // Destroy logical device (this invalidates all device objects)
        vkDestroyDevice(device, nullptr);
        // Clean up debug messenger if validation was enabled
        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }
        // Destroy platform-specific surface and instance
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        // Clean up window system resources
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    // === SWAPCHAIN RECREATION (WINDOW RESIZE AND RECOVERY HANDLING) ===
    // This function handles the complete recreation of the swapchain and all dependent 
    // Vulkan objects when the rendering surface changes. This is necessary because:
    //
    // 1. WINDOW RESIZING: When users resize the window, the swapchain dimensions
    //    must match the new surface size. The old swapchain becomes invalid.
    //
    // 2. SURFACE CHANGES: Display mode changes, fullscreen transitions, or 
    //    window state changes (minimize/maximize) can invalidate the swapchain.
    //
    // 3. DRIVER RECOVERY: Some platforms may mark swapchains as "suboptimal" 
    //    or "outdated" requiring recreation for optimal performance.
    //
    // The process involves:
    // - Waiting for all GPU work to complete (vkDeviceWaitIdle)
    // - Cleaning up old swapchain resources (framebuffers, image views, etc.)
    // - Creating a new swapchain with updated surface capabilities
    // - Recreating all dependent objects (image views, render pass, pipeline, etc.)
    // - Reallocating command buffers and resetting synchronization objects
    //
    // This is one of Vulkan's more complex requirements - unlike other graphics APIs
    // where window resizing is handled automatically, Vulkan applications must 
    // explicitly manage the entire swapchain lifecycle.
    void recreateSwapChain() {
        // Handle the case where window is minimized (size 0)
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        // Ensure GPU finishes before destroying resources
        vkDeviceWaitIdle(device);

        // Clean up all swapchain-dependent resources
        cleanupSwapChain();

        // Recreate the entire rendering pipeline with new dimensions
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandBuffers();

        // Reset frame synchronization tracking
        imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);
    }

    // === INSTANCE CREATION (STEP 1: VULKAN API ENTRY POINT) ===
    // The Vulkan instance is the foundational object that connects your application
    // to the Vulkan runtime. It represents your application's "connection" to Vulkan
    // and provides access to all other Vulkan functionality.
    //
    // KEY CONCEPTS:
    // - INSTANCE: Application-level object (not tied to specific GPU)
    // - VALIDATION LAYERS: Development tools for error checking and debugging
    // - EXTENSIONS: Additional Vulkan functionality beyond core specification
    // - DEBUG MESSENGER: Callback system for receiving Vulkan debug messages
    //
    // CREATION PROCESS:
    // 1. Verify validation layer support (development builds only)
    // 2. Define application information (name, version, engine info)
    // 3. Specify required extensions (platform integration, debug tools)
    // 4. Enable validation layers for development builds
    // 5. Create the instance object
    //
    // WHY THIS MATTERS:
    // - First Vulkan object that must be created
    // - Enables platform integration (window system connection)
    // - Provides access to physical devices (GPUs)
    // - Sets up debugging infrastructure for development
    void createInstance() {
        // Validation layers are development tools that intercept Vulkan calls
        // to check for errors, validate parameters, and track resource usage.
        // They're essential during development but typically disabled in release builds.
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        // Application info tells Vulkan about your application
        // This helps drivers optimize for specific applications
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;           // Required: Structure type identification
        appInfo.pApplicationName = "Hello Triangle";                  // Your application name
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);        // Your app version (major.minor.patch)
        appInfo.pEngineName = "No Engine";                            // Engine name (or "No Engine" for custom apps)
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);             // Engine version
        appInfo.apiVersion = VK_API_VERSION_1_0;                      // Vulkan API version you want to use

        // Instance creation info specifies what functionality your app needs
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;    // Required: Structure type
        createInfo.pApplicationInfo = &appInfo;                       // Link to application info above

        // Extensions provide platform integration and additional features
        // getRequiredExtensions() typically includes window system integration
        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        // Debug messenger setup for validation layers
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            // Enable validation layers for error checking
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            // Setup debug messenger to receive validation messages
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        } else {
            // No validation layers in release builds
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        // Finally create the Vulkan instance
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }
    // === DEBUG MESSENGER CONFIGURATION (STEP 2A: DEBUG SETUP HELPER) ===
    // Helper function that configures the parameters for Vulkan's debug messaging system.
    // This prepares the creation information for the debug messenger that will catch
    // and report Vulkan API usage errors during development.
    //
    // WHY THIS MATTERS: Vulkan doesn't automatically report errors like OpenGL does.
    // Validation layers + debug messenger are essential for catching mistakes during
    // development that would otherwise cause crashes or undefined behavior.
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};  // Initialize all fields to zero (required for Vulkan structs)
        
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        // ^ Required: Identifies this structure type to Vulkan
        
        // Specify which types of messages we want to receive:
        createInfo.messageSeverity = 
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |   // Verbose: Detailed diagnostic info
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |   // Warning: Suboptimal usage patterns  
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;      // Error: Invalid API usage
        // ^ Controls how serious the messages need to be for us to receive them
        
        // Specify which categories of messages we want:
        createInfo.messageType = 
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |       // General: Object creation/destruction
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |    // Validation: Spec violations, wrong params
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;    // Performance: Slow techniques, inefficiencies
        // ^ Controls what kinds of issues we want to be notified about
        
        createInfo.pfnUserCallback = debugCallback;
        // ^ Function pointer to our custom callback that will process and display messages
    }

    // === DEBUG MESSENGER CREATION (STEP 2: DEBUG INFRASTRUCTURE) ===
    // Creates the actual debug messenger object that starts monitoring Vulkan API calls
    // and reporting any issues through our callback function. This only runs in debug builds.
    //
    // CONNECTION TO PREVIOUS STEP: Uses the Vulkan instance created in Step 1
    // WHY THIS MATTERS: Without this, Vulkan errors silently corrupt your program or crash it
    void setupDebugMessenger() {
        // Skip debug setup in release builds for better performance
        if (!enableValidationLayers) return;

        // Prepare the configuration parameters for our debug messenger
        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        // Create the debug messenger using a helper function (Vulkan extension loading)
        // This starts the validation system monitoring all subsequent Vulkan calls
        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
        // ^ From this point forward, validation errors will appear in our debug callback
    }

    // === SURFACE CREATION (STEP 3: WINDOW SYSTEM INTEGRATION) ===
    // Creates a Vulkan surface object that represents the connection between Vulkan
    // rendering and the actual window on screen. This bridges the gap between
    // Vulkan's platform-agnostic API and your operating system's window system.
    //
    // CONNECTION TO PREVIOUS STEPS: Requires the Vulkan instance (Step 1) and GLFW window
    // WHY THIS MATTERS: Without a surface, you can't present rendered images to the screen
    void createSurface() {
        // Ask GLFW to create the appropriate platform-specific surface:
        // - Windows: Win32 surface
        // - Linux: X11 or Wayland surface  
        // - macOS: Metal surface (via MoltenVK)
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        // ^ surface now represents our window to Vulkan - essential for swapchain creation
    }

    // === PHYSICAL DEVICE SELECTION (STEP 4: GPU DISCOVERY) ===
    // Finds all Vulkan-capable GPUs in the system and selects the most suitable one.
    // Modern systems might have multiple GPUs (integrated + discrete, or multiple cards).
    //
    // CONNECTION TO PREVIOUS STEPS: Requires Vulkan instance (Step 1) to query devices
    // WHY THIS MATTERS: This is where we choose which actual hardware will do our rendering
    void pickPhysicalDevice() {
        // First, ask how many Vulkan-capable GPUs are available
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        // ^ deviceCount now contains the number of available GPUs

        // If no GPUs support Vulkan, we can't render anything
        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        // Get the actual list of physical device handles
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        // ^ devices now contains handles to all available GPUs

        // Check each GPU to see if it meets our requirements (graphics + present capability)
        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {  // Tests queue families, extensions, swapchain support
                physicalDevice = device;     // Found a good GPU!
                break;
            }
        }

        // If no suitable GPU was found, we can't proceed with rendering
        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
        // ^ physicalDevice now represents our chosen GPU hardware
    }

    // === LOGICAL DEVICE CREATION (STEP 5: GPU ACCESS SETUP) ===
    // Creates a logical device that represents your application's connection to the
    // physical GPU. This is where you specify which GPU features you want to use
    // and create the command queues that will execute your rendering commands.
    //
    // CONNECTION TO PREVIOUS STEPS: Requires physical device (Step 4) and surface (Step 3)
    // WHY THIS MATTERS: This is your primary interface for all GPU operations
    void createLogicalDevice() {
        // Find which queue families support the operations we need (graphics + presentation)
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        // ^ Queue families are groups of command queues with specific capabilities

        // Prepare to create queues from the required queue families
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        
        // Use a set to automatically handle the case where graphics and present
        // queues are the same family (common on most hardware)
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        // Queue priority affects scheduling when GPU is overloaded (0.0 to 1.0)
        float queuePriority = 1.0f;  // Maximum priority for our application
        
        // Create queue creation info for each unique queue family we need
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;      // Which family this queue belongs to
            queueCreateInfo.queueCount = 1;                      // We only need one queue per family
            queueCreateInfo.pQueuePriorities = &queuePriority;   // Priority for these queues
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // Specify which physical device features we want to enable
        // Currently empty - we're not using advanced features like geometry shaders, tessellation, etc.
        VkPhysicalDeviceFeatures deviceFeatures{};

        // Device creation info - this specifies all our requirements for the logical device
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        // Queue configuration - tell Vulkan which queues we want to create
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        // Enabled features - specify which GPU capabilities we want to use
        createInfo.pEnabledFeatures = &deviceFeatures;

        // Required device extensions - we need swapchain extension to present to window
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        // Validation layers (legacy - now inherited from instance in modern Vulkan)
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        // Finally create the logical device - this is where GPU access becomes available
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }
        // ^ device is now our primary interface to the GPU

        // Get the actual queue handles we'll use for command submission
        // These are the objects you submit rendering commands to
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);  // For drawing commands
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);    // For presenting images
        // ^ These queue handles are used throughout the rest of the application
    }

    // === SWAPCHAIN CREATION (STEP 6: RENDERING SURFACE SETUP) ===
    // Creates the swapchain, which is Vulkan's equivalent of a "backbuffer" - a
    // collection of images that can be rendered to and then presented to the screen.
    // The swapchain manages the presentation of rendered frames to the window.
    //
    // CONNECTION TO PREVIOUS STEPS: Requires logical device (Step 5), surface (Step 3),
    // and physical device (Step 4) to query surface capabilities
    // WHY THIS MATTERS: This is what allows you to actually display rendered images
    //
    // KEY CONCEPTS:
    // - SWAPCHAIN: Collection of images for rendering + presentation
    // - TRIPLE BUFFERING: Multiple images to prevent waiting (rendering while displaying)
    // - PRESENT MODE: How images are presented (immediate, vsync, mailbox)
    // - SURFACE FORMAT: Pixel format and color space for the images
    void createSwapChain() {
        // Query what swapchain features are supported by our GPU and surface
        // This tells us what options we have for format, present mode, image count, etc.
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
        // ^ Contains capabilities, supported formats, and present modes

        // Choose the best surface format (pixel format + color space)
        // e.g., VK_FORMAT_B8G8R8A8_SRGB for 8-bit BGRA with sRGB color space
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);

        // Choose the best presentation mode (how frames are displayed)
        // VK_PRESENT_MODE_FIFO_KHR = VSync (default, no tearing)
        // VK_PRESENT_MODE_IMMEDIATE_KHR = No VSync (fastest, potential tearing)
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);

        // Choose the swapchain dimensions (usually matches window size)
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
        // ^ This determines the resolution of our rendered images

        // Determine how many images we want in our swapchain
        // minImageCount + 1 gives us triple buffering (good for performance)
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        
        // Make sure we don't exceed the maximum supported image count
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }
        // ^ Typically 2-3 images for double/triple buffering

        // Configure the swapchain creation parameters
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;  // The window surface we want to present to

        createInfo.minImageCount = imageCount;                    // Number of images in swapchain
        createInfo.imageFormat = surfaceFormat.format;            // Pixel format (BGRA, RGBA, etc.)
        createInfo.imageColorSpace = surfaceFormat.colorSpace;    // Color space (usually sRGB)
        createInfo.imageExtent = extent;                          // Image dimensions (width/height)

        // For stereo 3D applications (almost always 1 for 2D rendering)
        createInfo.imageArrayLayers = 1;

        // How the images will be used - as color attachments for rendering
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        // Handle queue family sharing (when graphics and present queues are different)
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        // If graphics and present queues are from different families, we need concurrent access
        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;  // Multiple queues can use simultaneously
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;   // Single queue family owns the images
        }

        // Transform applied to images (usually none - use current transform)
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

        // Alpha blending with window system (opaque - no transparency)
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

        // Presentation mode (VSync, immediate, etc.)
        createInfo.presentMode = presentMode;

        // Enable clipping (don't render pixels that are obscured by other windows)
        createInfo.clipped = VK_TRUE;

        // For swapchain recreation - reference to old swapchain (none initially)
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        // Finally create the swapchain - this allocates the actual images
        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }
        // ^ swapChain now contains our collection of renderable images

        // Retrieve the actual swapchain images that were created
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);  // Query count first
        swapChainImages.resize(imageCount);                               // Resize our container
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());  // Get images
        // ^ swapChainImages now contains the actual VkImage handles we can render to

        // Store format and extent for later use in pipeline creation
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    // === IMAGE VIEW CREATION (STEP 7: IMAGE ACCESS SETUP) ===
    // Creates image views for each swapchain image. Image views are required to
    // tell Vulkan how to interpret images when they're used in rendering (as
    // textures, render targets, etc.). Swapchain images need views to be used
    // as color attachments in the render pass.
    //
    // CONNECTION TO PREVIOUS STEPS: Requires swapchain images (Step 6)
    // WHY THIS MATTERS: Without image views, you can't use images in rendering pipelines
    //
    // KEY CONCEPTS:
    // - IMAGE VIEW: Describes how to interpret an image (format, swizzling, subresources)
    // - SWIZZLING: Remapping color channels (r->b, g->g, b->r, a->a for example)
    // - ASPECT MASK: Which part of image to use (color, depth, stencil)
    void createImageViews() {
        // Create an image view for each swapchain image
        swapChainImageViews.resize(swapChainImages.size());

        // Process each swapchain image to create its corresponding view
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            // Configure the image view creation parameters
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            
            createInfo.image = swapChainImages[i];           // The image this view describes
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;     // 2D image (not cube map, 3D, etc.)
            createInfo.format = swapChainImageFormat;        // Same format as swapchain

            // Channel swizzling - how to map stored channels to shader channels
            // IDENTITY means no remapping (r->r, g->g, b->b, a->a)
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            // Specify which part of the image to use
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;  // Use color data
            createInfo.subresourceRange.baseMipLevel = 0;       // Start at first mip level
            createInfo.subresourceRange.levelCount = 1;         // Use only one mip level
            createInfo.subresourceRange.baseArrayLayer = 0;     // Start at first array layer
            createInfo.subresourceRange.layerCount = 1;         // Use only one array layer

            // Create the image view - this makes the image usable in rendering
            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
            // ^ Each image view can now be used as a color attachment in render passes
        }
        // ^ swapChainImageViews now contains views that connect swapchain images to the rendering pipeline
    }

    // === RENDER PASS CREATION (STEP 8: RENDERING WORKFLOW DEFINITION) ===
    // Creates a render pass that describes the rendering workflow - what attachments
    // (color, depth, etc.) will be used, how they'll be handled, and what operations
    // will be performed on them. The render pass defines the "contract" between
    // your rendering commands and the framebuffer attachments.
    //
    // CONNECTION TO PREVIOUS STEPS: Requires swapchain format (Step 6) to match attachment formats
    // WHY THIS MATTERS: This is where you define WHAT you're rendering to and HOW
    //
    // KEY CONCEPTS:
    // - ATTACHMENTS: The actual images being rendered to (color, depth, etc.)
    // - SUBPASSES: Groups of rendering operations (usually just one for simple cases)
    // - DEPENDENCIES: Synchronization between subpasses and external operations
    // - LAYOUT TRANSITIONS: How image memory layouts change during rendering
    void createRenderPass() {
        // Define the color attachment (our swapchain image)
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;              // Must match swapchain image format
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;            // No multisampling (1 sample per pixel)
        
        // What to do with attachment contents at the start of render pass
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;       // Clear to black/transparent
        // What to do with attachment contents at the end of render pass  
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;     // Keep the rendered image
        
        // Stencil operations (we're not using stencil, so don't care)
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        
        // Image layout transitions (how GPU memory is organized)
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;         // Don't care about previous contents
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;     // Ready for presentation to screen

        // Reference to the attachment for use in subpass
        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;  // Index of attachment in pAttachments array
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;  // Layout during subpass

        // Define the subpass (group of rendering operations)
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;  // This is a graphics subpass (not compute)
        subpass.colorAttachmentCount = 1;                             // Using one color attachment
        subpass.pColorAttachments = &colorAttachmentRef;              // Reference to our color attachment

        // Define dependency to ensure proper synchronization
        // This ensures rendering commands wait for the right time to access attachments
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;                 // External operations (before render pass)
        dependency.dstSubpass = 0;                                   // Our subpass (index 0)
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;  // Wait for color attachment setup
        dependency.srcAccessMask = 0;                                // No prior access to wait for
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;  // When we want to write colors
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;          // We want to write to color attachment

        // Create the render pass with all our configuration
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;                         // One attachment (color)
        renderPassInfo.pAttachments = &colorAttachment;             // Our color attachment description
        renderPassInfo.subpassCount = 1;                            // One subpass
        renderPassInfo.pSubpasses = &subpass;                       // Our subpass definition
        renderPassInfo.dependencyCount = 1;                         // One dependency
        renderPassInfo.pDependencies = &dependency;                 // Our dependency definition

        // Finally create the render pass - this defines our rendering workflow
        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
        // ^ renderPass now defines how our triangle rendering will work:
        //   1. Clear the screen to a known color
        //   2. Render our triangle to the color attachment
        //   3. Prepare the image for presentation to the screen
    }

    // === GRAPHICS PIPELINE CREATION (STEP 9: RENDERING PIPELINE SETUP) ===
    // Creates the graphics pipeline that defines HOW vertices are processed and
    // HOW pixels are shaded. This is the heart of rendering - it combines shaders,
    // vertex input configuration, rasterization settings, and blending to create
    // the complete rendering pipeline.
    //
    // CONNECTION TO PREVIOUS STEPS: Requires render pass (Step 8), swapchain extent (Step 6)
    // WHY THIS MATTERS: This is what actually defines how your 3D geometry becomes pixels on screen
    //
    // KEY CONCEPTS:
    // - SHADERS: Programs that run on GPU (vertex shader + fragment shader)
    // - VERTEX INPUT: How vertex data is structured and fed to the pipeline
    // - ASSEMBLY: How vertices are grouped into primitives (triangles, lines, points)
    // - VIEWPORT: How 3D coordinates map to 2D screen space
    // - RASTERIZATION: Converting primitives to fragments (pixels)
    // - BLENDING: How new pixels combine with existing pixels
    void createGraphicsPipeline() {
        // === SHADER LOADING AND MODULE CREATION ===
        // Load precompiled SPIR-V shader bytecode from files
        auto vertShaderCode = readFile("vertex_shader.vert.spv");    // Vertex shader: processes vertices
        auto fragShaderCode = readFile("fragment_shader.frag.spv");  // Fragment shader: shades pixels

        // Create Vulkan shader modules from the bytecode
        // Shader modules are containers for shader code - they don't execute yet
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        // Configure vertex shader stage
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;     // This is the vertex shader
        vertShaderStageInfo.module = vertShaderModule;              // The shader code module
        vertShaderStageInfo.pName = "main";                         // Entry point function name

        // Configure fragment shader stage
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;   // This is the fragment shader
        fragShaderStageInfo.module = fragShaderModule;              // The shader code module
        fragShaderStageInfo.pName = "main";                         // Entry point function name

        // Array of all shader stages (vertex + fragment for basic rendering)
        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        // === VERTEX INPUT CONFIGURATION ===
        // Define how vertex data is structured in memory
        // For this simple example (colored triangle), we're not using vertex buffers
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;     // No vertex buffers
        vertexInputInfo.vertexAttributeDescriptionCount = 0;   // No vertex attributes
        // ^ This means our vertex shader must generate vertex positions directly

        // === PRIMITIVE ASSEMBLY ===
        // Define how vertices are assembled into primitives (triangles, lines, points)
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;  // Triangles from every 3 vertices
        inputAssembly.primitiveRestartEnable = VK_FALSE;               // Don't restart primitive strips

        // === VIEWPORT AND SCISSOR CONFIGURATION ===
        // Define how 3D coordinates map to 2D screen space
        VkViewport viewport{};
        viewport.x = 0.0f;                              // Viewport left edge
        viewport.y = 0.0f;                              // Viewport top edge
        viewport.width = (float) swapChainExtent.width; // Full swapchain width
        viewport.height = (float) swapChainExtent.height; // Full swapchain height
        viewport.minDepth = 0.0f;                       // Near clipping plane
        viewport.maxDepth = 1.0f;                       // Far clipping plane

        // Scissor test defines which pixels can be modified
        VkRect2D scissor{};
        scissor.offset = {0, 0};                        // Scissor rectangle top-left
        scissor.extent = swapChainExtent;               // Scissor rectangle size (full screen)

        // Combine viewport and scissor into viewport state
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;                // One viewport
        viewportState.pViewports = &viewport;           // Our viewport configuration
        viewportState.scissorCount = 1;                 // One scissor rectangle
        viewportState.pScissors = &scissor;             // Our scissor configuration

        // === RASTERIZATION SETTINGS ===
        // Define how primitives are converted to fragments (pixels)
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;         // Don't clamp depth values
        rasterizer.rasterizerDiscardEnable = VK_FALSE;  // Don't discard primitives before fragment stage
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;  // Fill triangles (not wireframe/points)
        rasterizer.lineWidth = 1.0f;                    // Line width for wireframe (1.0 is required minimum)
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;    // Cull back faces for performance
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE; // Define front face winding order
        rasterizer.depthBiasEnable = VK_FALSE;          // Don't apply depth bias

        // === MULTISAMPLING CONFIGURATION ===
        // Define anti-aliasing settings (off for this simple example)
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;           // No sample shading
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT; // 1 sample per pixel (no MSAA)

        // === COLOR BLENDING ===
        // Define how new pixel colors combine with existing framebuffer contents
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = 
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | 
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;  // Enable writing to all color channels
        colorBlendAttachment.blendEnable = VK_FALSE;              // No blending (overwrite existing pixels)

        // Global color blending settings
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;                   // No logical operations
        colorBlending.logicOp = VK_LOGIC_OP_COPY;                 // (ignored when logicOpEnable = false)
        colorBlending.attachmentCount = 1;                        // One color attachment
        colorBlending.pAttachments = &colorBlendAttachment;       // Our blending configuration
        colorBlending.blendConstants[0] = 0.0f;                   // Blend constants (unused)
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        // === PIPELINE LAYOUT CREATION ===
        // Define the pipeline layout (uniforms, textures, push constants)
        // Empty for this simple example - no uniforms or textures
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0;          // No descriptor sets
        pipelineLayoutInfo.pushConstantRangeCount = 0;  // No push constants

        // Create the pipeline layout - this defines the "interface" for shader parameters
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        // === FINAL PIPELINE CREATION ===
        // Combine all pipeline states into one graphics pipeline
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;                           // Two shader stages (vertex + fragment)
        pipelineInfo.pStages = shaderStages;                   // Our shader stage configurations
        pipelineInfo.pVertexInputState = &vertexInputInfo;     // Vertex input configuration
        pipelineInfo.pInputAssemblyState = &inputAssembly;     // Primitive assembly configuration
        pipelineInfo.pViewportState = &viewportState;          // Viewport/scissor configuration
        pipelineInfo.pRasterizationState = &rasterizer;        // Rasterization configuration
        pipelineInfo.pMultisampleState = &multisampling;       // Multisampling configuration
        pipelineInfo.pColorBlendState = &colorBlending;        // Color blending configuration
        pipelineInfo.layout = pipelineLayout;                  // Pipeline layout (uniforms, etc.)
        pipelineInfo.renderPass = renderPass;                  // Render pass this pipeline is compatible with
        pipelineInfo.subpass = 0;                              // Subpass index (first and only subpass)
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;      // No pipeline derivation

        // FINALLY create the graphics pipeline - this compiles all our settings
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
        // ^ graphicsPipeline is now ready to render our triangle!

        // Clean up temporary shader modules (no longer needed after pipeline creation)
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    // === FRAMEBUFFER CREATION (STEP 10: RENDER TARGET SETUP) ===
    // Creates framebuffers that connect image views to the render pass. A framebuffer
    // specifies which actual images will be used for each attachment when executing
    // a render pass. Each swapchain image needs its own framebuffer.
    //
    // CONNECTION TO PREVIOUS STEPS: Requires image views (Step 7) and render pass (Step 8)
    // WHY THIS MATTERS: This links your actual render targets (swapchain images) to the
    // rendering workflow defined in the render pass
    void createFramebuffers() {
        // Create one framebuffer for each swapchain image view
        swapChainFramebuffers.resize(swapChainImageViews.size());

        // Process each swapchain image to create its corresponding framebuffer
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            // Define which attachments this framebuffer will use
            // For simple rendering, just the color attachment (swapchain image)
            VkImageView attachments[] = {
                swapChainImageViews[i]  // The actual image view for this framebuffer
            };

            // Configure framebuffer creation parameters
            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;              // Must match the render pass
            framebufferInfo.attachmentCount = 1;                  // One attachment (color)
            framebufferInfo.pAttachments = attachments;           // Our attachment array
            framebufferInfo.width = swapChainExtent.width;        // Match swapchain dimensions
            framebufferInfo.height = swapChainExtent.height;      // Match swapchain dimensions
            framebufferInfo.layers = 1;                           // Single layer (not array texture)

            // Create the framebuffer - this connects the image view to the render pass
            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
            // ^ Each framebuffer can now be used to render to its corresponding swapchain image
        }
        // ^ swapChainFramebuffers now contains framebuffers ready for rendering commands
    }

    // === COMMAND POOL CREATION (STEP 11: COMMAND ALLOCATION SETUP) ===
    // Creates a command pool that serves as a memory allocator for command buffers.
    // Command pools are associated with specific queue families and manage the memory
    // used by command buffers allocated from them.
    //
    // CONNECTION TO PREVIOUS STEPS: Requires queue family information (Step 4/5)
    // WHY THIS MATTERS: All command buffers must come from a command pool, and the
    // pool determines which queue family can use the command buffers
    void createCommandPool() {
        // Find which queue family will be used for graphics commands
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        // Configure command pool creation parameters
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();  // Graphics queue family
        // poolInfo.flags = 0; // Default: buffers can only be reset individually

        // Create the command pool - this allocates memory for command buffers
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
        // ^ commandPool can now allocate command buffers for graphics queue
    }

    // === COMMAND BUFFER CREATION AND RECORDING (STEP 12: RENDERING COMMANDS) ===
    // Allocates command buffers from the command pool and records the actual
    // rendering commands that will be submitted to the GPU. This is where you
    // define WHAT will be rendered each frame.
    //
    // CONNECTION TO PREVIOUS STEPS: Requires command pool (Step 11), framebuffers (Step 10),
    // graphics pipeline (Step 9), and render pass (Step 8)
    // WHY THIS MATTERS: This is where you record the specific draw commands that
    // make your triangle (or any geometry) actually appear on screen
    void createCommandBuffers() {
        // Create one command buffer for each framebuffer (one per swapchain image)
        commandBuffers.resize(swapChainFramebuffers.size());

        // === COMMAND BUFFER ALLOCATION ===
        // Configure how to allocate the command buffers
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;                          // Pool to allocate from
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;            // Primary buffers (can be submitted directly)
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size(); // One per framebuffer

        // Allocate the command buffers from the pool
        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
        // ^ commandBuffers now contains handles to writable command buffers

        // === COMMAND BUFFER RECORDING ===
        // Record rendering commands into each command buffer
        for (size_t i = 0; i < commandBuffers.size(); i++) {
            // Configure how to begin recording commands
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            // beginInfo.flags = 0; // Default: can only record once (we'll recreate on resize)

            // Start recording commands into this command buffer
            if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("failed to begin recording command buffer!");
            }

            // === RENDER PASS EXECUTION ===
            // Configure the render pass execution parameters
            VkRenderPassBeginInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = renderPass;                    // The render pass to execute
            renderPassInfo.framebuffer = swapChainFramebuffers[i];     // Which framebuffer to render to
            renderPassInfo.renderArea.offset = {0, 0};                 // Render area top-left
            renderPassInfo.renderArea.extent = swapChainExtent;        // Render area size (full screen)

            // Define clear values for attachments (clear to black with full alpha)
            VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};   // Clear color: opaque black
            renderPassInfo.clearValueCount = 1;                       // One clear value
            renderPassInfo.pClearValues = &clearColor;                // Our clear value

            // Begin the render pass - this starts the rendering workflow
            vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

                // === ACTUAL RENDERING COMMANDS ===
                // Bind the graphics pipeline that defines how to render
                vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

                // Draw 3 vertices (a triangle) with 1 instance, starting at vertex 0, instance 0
                // Since we have no vertex buffers, the vertex shader must generate positions
                vkCmdDraw(commandBuffers[i], 3, 1, 0, 0);
                // ^ This is the actual draw call that renders our triangle!

            // End the render pass - this finishes the rendering workflow
            vkCmdEndRenderPass(commandBuffers[i]);

            // Finish recording commands into this command buffer
            if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer!");
            }
            // ^ commandBuffers[i] now contains all commands needed to render one frame
        }
        // ^ All command buffers are now recorded and ready for submission
    }

    // === SYNCHRONIZATION OBJECTS CREATION (STEP 13: FRAME TIMING SETUP) ===
    // Creates the synchronization primitives needed to coordinate between
    // CPU and GPU, and between different stages of the rendering pipeline.
    // Vulkan is heavily multi-threaded and asynchronous, so explicit
    // synchronization is required to prevent race conditions.
    //
    // CONNECTION TO PREVIOUS STEPS: Requires logical device (Step 5)
    // WHY THIS MATTERS: Without proper synchronization, you'll get crashes,
    // corrupted frames, or GPU stalls. This enables triple buffering workflow.
    //
    // KEY CONCEPTS:
    // - SEMAPHORES: GPU-GPU synchronization (signal from one queue, wait on another)
    // - FENCES: CPU-GPU synchronization (CPU waits for GPU completion)
    // - TRIPLE BUFFERING: Multiple frames in flight simultaneously for performance
    void createSyncObjects() {
        // Resize containers to hold synchronization objects for each frame in flight
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);    // GPU signals when image is ready
        // renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT); // ← OLD LINE: Remove or comment out
        renderFinishedSemaphores.resize(swapChainImages.size()); // ← NEW: One per swapchain image
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);              // CPU waits for frame completion
        imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE); // Tracks which fence owns each image

        // Configure semaphore creation (used for GPU-GPU synchronization)
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        // Semaphores have no additional parameters - they're just binary signals

        // Configure fence creation (used for CPU-GPU synchronization)
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // Start signaled so first wait doesn't block

        // Create synchronization objects for each frame that can be in flight
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            // Create semaphore for signaling when swapchain image is acquired
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                // Create fence for CPU to wait on frame completion
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
                }
        }

        // Create one render finished semaphore per swapchain image (not per in-flight frame)
        // This prevents reuse of a semaphore that's still in use by vkQueuePresentKHR
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create render finished semaphore!");
            }
        }
        // ^ Now we have all synchronization primitives needed for smooth triple buffering
    }

    // === FRAME RENDERING AND PRESENTATION (STEP 14: MAIN RENDER LOOP) ===
    // The core rendering function that executes each frame. This implements the
    // complete rendering pipeline: acquire image → render → present image.
    // This function is called repeatedly in the main loop to render animation.
    //
    // CONNECTION TO PREVIOUS STEPS: Requires ALL previous steps (command buffers,
    // synchronization objects, swapchain, queues, etc.)
    // WHY THIS MATTERS: This is the actual "game loop" that makes things appear on screen
    //
    // RENDERING FLOW:
    // 1. Wait for previous frame completion (CPU-GPU sync)
    // 2. Acquire next swapchain image (GPU tells us which image to use)
    // 3. Wait for image to be ready (if still in use from previous frames)
    // 4. Submit rendering commands (GPU does the actual rendering)
    // 5. Present rendered image to screen (display the result)
    void drawFrame() {
        // === FRAME SYNCHRONIZATION: WAIT FOR PREVIOUS FRAME ===
        // Wait for the fence to be signaled, indicating this frame slot is available
        // This prevents overwriting command buffers that might still be executing
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        // ^ UINT64_MAX = wait forever (no timeout)

        // === SWAPCHAIN IMAGE ACQUISITION ===
        // Ask Vulkan which swapchain image we can render to next
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
                                                imageAvailableSemaphores[currentFrame],  // Signal this when ready
                                                VK_NULL_HANDLE,                          // No fence for this operation
                                                &imageIndex);                            // Returns the image index

        // Handle swapchain issues (out of date, suboptimal)
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            // Swapchain is invalid (window resized, display changed) - recreate it
            recreateSwapChain();
            return;  // Skip rendering this frame, try again next iteration
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            // Unexpected error occurred
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // === IMAGE USAGE SYNCHRONIZATION ===
        // Ensure this specific image isn't still being used by a previous frame
        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
            // Wait for the fence that owns this image to be signaled
            vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        }
        // Mark this image as being used by the current frame's fence
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        // === COMMAND SUBMISSION TO GPU ===
        // Submit our pre-recorded command buffer to the graphics queue for execution
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        // Define what to wait for before executing commands
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};  // Wait for image acquisition
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};  // Wait at color attachment stage
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;        // Semaphores to wait for
        submitInfo.pWaitDstStageMask = waitStages;          // Pipeline stages where waiting occurs

        // Specify which command buffers to execute
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex];  // Use command buffer for this image

        // Define what to signal when commands complete
        // ✅ Use imageIndex to index renderFinishedSemaphores, not currentFrame
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[imageIndex]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;    // Semaphores to signal

        // Reset the fence before submitting work (it was signaled from previous frame)
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        // Submit the command buffer to the graphics queue for execution
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }
        // ^ GPU now starts executing our rendering commands asynchronously

        // === IMAGE PRESENTATION TO SCREEN ===
        // Present the rendered image to the screen (display it in the window)
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        // Wait for rendering to complete before presenting
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;     // Wait for rendering completion

        // Specify which swapchain to present to
        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;               // Our swapchain
        presentInfo.pImageIndices = &imageIndex;            // Which image to present

        // Present the image to the screen
        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        // Handle presentation issues
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            // Swapchain needs recreation (resize, display change, or suboptimal)
            framebufferResized = false;
            recreateSwapChain();  // Recreate and try again next frame
        } else if (result != VK_SUCCESS) {
            // Unexpected presentation error
            throw std::runtime_error("failed to present swap chain image!");
        }

        // === FRAME MANAGEMENT: ADVANCE TO NEXT FRAME ===
        // Move to the next frame slot for triple buffering
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        // ^ This creates the circular buffer pattern: 0, 1, 0, 1, 0, 1... (for MAX_FRAMES_IN_FLIGHT = 2)
    }

    // === SHADER MODULE CREATION (HELPER FOR STEP 9: PIPELINE CREATION) ===
    // Creates a Vulkan shader module from SPIR-V bytecode. Shader modules are
    // containers for shader code that can be referenced by pipeline creation.
    // This is a helper function used during graphics pipeline creation.
    //
    // WHY THIS MATTERS: Vulkan requires pre-compiled SPIR-V shaders, unlike
    // OpenGL's GLSL which gets compiled at runtime. This function wraps the
    // low-level shader module creation process.
    VkShaderModule createShaderModule(const std::vector<char>& code) {
        // Configure shader module creation parameters
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();                          // Size of shader bytecode
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data()); // Pointer to bytecode (must be uint32_t*)

        // Create the shader module container
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }
        // ^ shaderModule now contains the compiled shader code, ready for pipeline use

        return shaderModule;  // Caller is responsible for destroying this module
    }

    // === SWAPCHAIN FORMAT SELECTION (HELPER FOR STEP 6: SWAPCHAIN CREATION) ===
    // Chooses the best surface format (pixel format + color space) for the swapchain.
    // Different GPUs support different formats, so we need to select a good one.
    //
    // WHY THIS MATTERS: The format determines how pixels are stored and displayed.
    // VK_FORMAT_B8G8R8A8_SRGB is widely supported and provides good color quality.
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        // Look for our preferred format: 8-bit BGRA with sRGB color space
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && 
                availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;  // Found our ideal format!
            }
        }

        // If preferred format isn't available, just use the first supported format
        return availableFormats[0];
        // ^ In a real application, you might want to score formats and pick the best
    }

    // === SWAPCHAIN PRESENT MODE SELECTION (HELPER FOR STEP 6: SWAPCHAIN CREATION) ===
    // Chooses the best presentation mode for displaying frames. This controls
    // how rendered images are presented to the screen (VSync behavior).
    //
    // PRESENT MODES:
    // - VK_PRESENT_MODE_IMMEDIATE_KHR: No VSync, fastest but can cause tearing
    // - VK_PRESENT_MODE_FIFO_KHR: VSync, waits for display refresh (default)
    // - VK_PRESENT_MODE_MAILBOX_KHR: Triple buffering, replaces queued frames
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        // Look for mailbox mode (triple buffering) for smoothest performance
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;  // Found triple buffering!
            }
        }

        // Fall back to FIFO (VSync) if mailbox isn't available
        // FIFO is guaranteed to be available and prevents tearing
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    // === SWAPCHAIN EXTENT SELECTION (HELPER FOR STEP 6: SWAPCHAIN CREATION) ===
    // Determines the swapchain image dimensions. Usually matches window size,
    // but some platforms have special requirements.
    //
    // WHY THIS MATTERS: Images must be the right size to fill the window properly.
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        // Check if the surface has a preferred extent (common on some mobile/embedded platforms)
        if (capabilities.currentExtent.width != UINT32_MAX) {
            return capabilities.currentExtent;  // Use the surface's preferred size
        } else {
            // Get actual window size and clamp it to supported range
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            // Clamp to supported minimum and maximum extents
            actualExtent.width = std::max(capabilities.minImageExtent.width, 
                                        std::min(capabilities.maxImageExtent.width, actualExtent.width));
            actualExtent.height = std::max(capabilities.minImageExtent.height, 
                                         std::min(capabilities.maxImageExtent.height, actualExtent.height));

            return actualExtent;  // Return clamped window size
        }
    }

    // === SWAPCHAIN CAPABILITIES QUERY (HELPER FOR STEP 6: SWAPCHAIN CREATION) ===
    // Queries all supported swapchain features for a given physical device and surface.
    // This information is needed to create a compatible swapchain.
    //
    // WHY THIS MATTERS: Different GPUs and window systems support different
    // swapchain configurations. This function discovers what's possible.
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        // Query basic surface capabilities (min/max image count, extent limits, etc.)
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        // Query supported surface formats (pixel formats and color spaces)
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        // Query supported presentation modes (VSync behavior)
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;  // Contains all information needed to create a compatible swapchain
    }
    // === DEVICE SUITABILITY CHECK (HELPER FOR STEP 4: GPU SELECTION) ===
    // Determines if a physical device (GPU) is suitable for our application.
    // A device is suitable if it supports all required features: graphics queue,
    // presentation queue, required extensions, and swapchain support.
    //
    // WHY THIS MATTERS: Modern systems may have multiple GPUs (integrated + discrete)
    // or devices that don't support graphics (compute-only cards). We need to
    // find one that can actually render to our window.
    //
    // CHECKS PERFORMED:
    // 1. Queue family support (graphics + presentation)
    // 2. Device extension support (swapchain extension)
    // 3. Swapchain capability (formats + present modes)
    bool isDeviceSuitable(VkPhysicalDevice device) {
        // Check 1: Find queue families that support graphics and presentation
        QueueFamilyIndices indices = findQueueFamilies(device);
        // ^ We need both graphics (for rendering) and present (for displaying)

        // Check 2: Verify the device supports all required extensions
        bool extensionsSupported = checkDeviceExtensionSupport(device);
        // ^ Essential for swapchain functionality

        // Check 3: Verify swapchain support (only check if extensions are supported)
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            // Query what swapchain features this device supports with our surface
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            // Device is adequate if it supports at least one format and present mode
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        // Device is suitable only if ALL requirements are met
        return indices.isComplete() && extensionsSupported && swapChainAdequate;
        // ^ indices.isComplete() means both graphics and present queues were found
    }

    // === DEVICE EXTENSION SUPPORT CHECK (HELPER FOR DEVICE SUITABILITY) ===
    // Verifies that a physical device supports all the extensions our application needs.
    // This is crucial because extensions provide additional Vulkan functionality
    // beyond the core specification.
    //
    // WHY THIS MATTERS: The swapchain extension is required for presenting images
    // to the window. Some GPUs might not support it (old drivers, compute cards).
    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        // Query how many device extensions are available
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        // ^ nullptr means "all extensions", &extensionCount gets the count

        // Get the actual list of available extensions
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
        // ^ availableExtensions now contains all supported extensions

        // Create a set of required extensions for easy lookup/removal
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
        // ^ deviceExtensions contains VK_KHR_SWAPCHAIN_EXTENSION_NAME

        // Remove each available extension from our required set
        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
            // ^ If an extension is available, remove it from required set
        }

        // If requiredExtensions is empty, all required extensions were found
        return requiredExtensions.empty();
        // ^ True means all required extensions are supported by this device
    }

    // === QUEUE FAMILY DISCOVERY (HELPER FOR MULTIPLE STEPS) ===
    // Finds the queue families supported by a physical device and identifies
    // which ones support graphics operations and presentation to our surface.
    // Queue families are groups of queues with specific capabilities.
    //
    // WHY THIS MATTERS: 
    // - Graphics queues: For submitting rendering commands
    // - Present queues: For displaying rendered images to the window
    // Some hardware has separate families, some combine them.
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;  // Will store the queue family indices we find

        // Query how many queue families this device supports
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        // ^ &queueFamilyCount gets the count, nullptr means "just count them"

        // Get the actual queue family properties
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
        // ^ queueFamilies now contains information about each queue family

        // Check each queue family for the capabilities we need
        int i = 0;  // Queue family index
        for (const auto& queueFamily : queueFamilies) {
            // Check if this queue family supports graphics operations
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;  // Found graphics queue family!
            }

            // Check if this queue family supports presentation to our surface
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            // ^ This is a surface-specific query - not all queues can present to all surfaces

            if (presentSupport) {
                indices.presentFamily = i;  // Found presentation queue family!
            }

            // Early exit: if we found both families, we're done
            if (indices.isComplete()) {
                break;  // No need to check remaining queue families
            }

            i++;  // Move to next queue family
        }

        return indices;  // Contains indices of suitable queue families (or nullopt if not found)
        // ^ This gets used in logical device creation and swapchain setup
    }

    // === REQUIRED EXTENSIONS QUERY (HELPER FOR STEP 1: INSTANCE CREATION) ===
    // Gets the list of instance extensions required for this application to function.
    // This includes platform-specific extensions for window system integration
    // (provided by GLFW) and optional debug extensions for development.
    //
    // WHY THIS MATTERS: 
    // - Platform extensions: Enable Vulkan to communicate with the window system
    // - Debug extensions: Enable validation layer messaging during development
    // Without these, you can't create windows or get debug feedback.
    std::vector<const char*> getRequiredExtensions() {
        // Ask GLFW what platform-specific extensions are needed for window creation
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        // ^ On Windows this returns "VK_KHR_surface" and "VK_KHR_win32_surface"
        // ^ On Linux this returns "VK_KHR_surface" and "VK_KHR_xcb_surface" (or similar)

        // Create a vector containing the GLFW-required extensions
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        // ^ This provides window system integration capability

        // Add debug extension if validation layers are enabled (development only)
        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            // ^ This enables the debug messenger functionality
        }

        return extensions;  // Complete list of required instance extensions
    }

    // === VALIDATION LAYER SUPPORT CHECK (HELPER FOR STEP 1: INSTANCE CREATION) ===
    // Verifies that all requested validation layers are available on this system.
    // Validation layers are development tools that intercept Vulkan API calls
    // to check for errors and provide debugging information.
    //
    // WHY THIS MATTERS: Validation layers are essential for catching Vulkan
    // programming errors during development, but they may not be available
    // on all systems (especially release builds or minimal installations).
    bool checkValidationLayerSupport() {
        // Query how many validation layers are available on this system
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        // ^ &layerCount gets the count, nullptr means "just count them"

        // Get the actual list of available validation layers
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
        // ^ availableLayers now contains information about all available layers

        // Check if each requested validation layer is available
        for (const char* layerName : validationLayers) {
            // validationLayers contains "VK_LAYER_KHRONOS_validation"
            bool layerFound = false;

            // Search through available layers for this requested layer
            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;  // Found the requested layer!
                    break;
                }
            }

            // If any required layer is missing, validation isn't supported
            if (!layerFound) {
                return false;  // Missing a required validation layer
            }
        }

        return true;  // All requested validation layers are available
    }

    // === FILE READING UTILITY (HELPER FOR STEP 9: SHADER LOADING) ===
    // Reads a binary file (typically compiled SPIR-V shader) into a byte vector.
    // This is used to load pre-compiled shaders from disk.
    //
    // WHY THIS MATTERS: Vulkan requires shaders to be pre-compiled to SPIR-V
    // bytecode. This function loads that bytecode so it can be turned into
    // Vulkan shader modules.
    static std::vector<char> readFile(const std::string& filename) {
        // Open file in binary mode at the end to get file size
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        // ^ std::ios::ate = "at end" - file pointer starts at end for size calculation

        // Check if file opened successfully
        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        // Get file size from current position (which is at the end)
        size_t fileSize = (size_t) file.tellg();
        
        // Create buffer of appropriate size
        std::vector<char> buffer(fileSize);

        // Seek back to beginning and read entire file
        file.seekg(0);                    // Go back to start of file
        file.read(buffer.data(), fileSize); // Read all data into buffer

        // Close the file
        file.close();

        return buffer;  // Vector containing entire file contents
    }

    // === DEBUG CALLBACK FUNCTION (HELPER FOR STEP 2: DEBUG MESSAGING) ===
    // The callback function that receives and processes validation layer messages.
    // This is where validation errors, warnings, and debug information are displayed.
    //
    // WHY THIS MATTERS: Without this callback, validation layer messages would
    // be silently discarded. This is your primary debugging tool in Vulkan.
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,    // How serious the message is
        VkDebugUtilsMessageTypeFlagsEXT messageType,              // What type of message
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, // Message details
        void* pUserData) {                                        // Custom user data (unused here)
        
        // Print the validation message to standard error
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        // ^ pCallbackData->pMessage contains the actual error/warning text

        return VK_FALSE;  // VK_FALSE = don't abort execution
        // ^ Return VK_TRUE only if you want to force a crash on validation errors
    }
};

// === APPLICATION ENTRY POINT ===
// The main function that creates and runs the Vulkan application.
// This is the standard C++ entry point where execution begins.
//
// WHY THIS MATTERS: This is where your Vulkan journey starts! The entire
// complex initialization pipeline we've been exploring is triggered from here.
//
// EXECUTION FLOW:
// 1. Create application object (HelloTriangleApplication)
// 2. Call app.run() which triggers the complete 14-step initialization
// 3. Enter main rendering loop (drawFrame() called repeatedly)
// 4. Handle any exceptions and exit cleanly
//
// ERROR HANDLING: Vulkan operations can fail for many reasons (no GPU,
// missing drivers, invalid parameters). All errors are caught here and
// reported to the user before clean shutdown.
int main() {
    // Create the application object - this doesn't start anything yet
    HelloTriangleApplication app;

    try {
        // This is where the magic happens! Calling run() triggers:
        // - Window creation (GLFW)
        // - Vulkan instance creation (14-step initialization pipeline)
        // - Main rendering loop (drawFrame() in infinite loop)
        // - Clean shutdown when window is closed
        app.run();
        
        // If we reach here, application exited normally (window closed)
        
    } catch (const std::exception& e) {
        // Catch any standard exceptions (runtime_error, etc.)
        // This handles all the "throw std::runtime_error()" calls throughout our code
        std::cerr << e.what() << std::endl;  // Print error message
        return EXIT_FAILURE;                 // Exit with error code
    }

    // Normal exit - application closed successfully
    return EXIT_SUCCESS;  // Exit with success code
}
