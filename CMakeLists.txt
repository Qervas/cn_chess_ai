cmake_minimum_required(VERSION 3.16)

project(ChessAI LANGUAGES CXX CUDA)

if(WIN32)
    # Set Qt path for Windows
    set(CMAKE_PREFIX_PATH "C:/Qt/6.8.0/msvc2022_64")
    
    # Set CUDA paths for Windows
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe")
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
    
    # Set cuDNN paths for Windows
    set(CUDNN_ROOT_DIR "C:/Program Files/NVIDIA/CUDNN/v9.5")
    set(CUDNN_LIB_DIR "${CUDNN_ROOT_DIR}/lib/12.6/x64")
    set(CUDNN_INCLUDE_DIR "${CUDNN_ROOT_DIR}/include/12.6")
else()
    # UNIX configurations
    # set(CMAKE_PREFIX_PATH "/path/to/your/qt")
    # Usually doesn't need to be set specifically if in the system PATH
endif()

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Find cuDNN Library
if(WIN32)
    # Specify the exact name of the cuDNN library file
    # Adjust the name based on your installation
    find_library(CUDNN_LIBRARY 
        NAMES cudnn cudnn64_9 cudnn.lib cudnn64_9.lib
        HINTS "${CUDNN_LIB_DIR}"
        PATH_SUFFIXES lib x64
        REQUIRED
    )
else()
    find_library(CUDNN_LIBRARY cudnn
        HINTS ${CUDAToolkit_LIBRARY_DIR}
        REQUIRED)
endif()

if(NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN library not found in ${CUDNN_LIB_DIR}")
else()
    message(STATUS "Found cuDNN library: ${CUDNN_LIBRARY}")
endif()

# Find cuDNN Include Directory
if(WIN32)
    find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS "${CUDNN_INCLUDE_DIR}"
        REQUIRED
    )
else()
    find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS}
        REQUIRED)
endif()

if(NOT CUDNN_INCLUDE_DIR)
    message(FATAL_ERROR "cuDNN include directory not found in ${CUDNN_INCLUDE_DIR}")
else()
    message(STATUS "Found cuDNN include directory: ${CUDNN_INCLUDE_DIR}")
endif()

# Qt Configuration
find_package(Qt6 COMPONENTS Widgets Multimedia Charts REQUIRED)

# Set C++ and CUDA Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# CUDA Architectures and Flags
set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -arch=sm_86")

# Source and Header Files
file(GLOB SOURCES "src/*.cpp" "src/*.cu")
file(GLOB HEADERS "include/*.h" "include/*.cuh")

# Qt Resources
qt6_add_resources(RESOURCES resources.qrc)

# Executable
add_executable(ChessAI
    ${SOURCES}
    ${HEADERS}
    ${RESOURCES}
)

if (MSVC)
	target_compile_options(ChessAI PRIVATE /Zc:__cplusplus)
endif()


# CUDA Separable Compilation
set_target_properties(ChessAI PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include Directories
target_include_directories(ChessAI PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDNN_INCLUDE_DIR}
)

# Link Libraries
target_link_libraries(ChessAI PRIVATE
    Qt6::Widgets
    Qt6::Multimedia
    Qt6::Charts
    Qt6::Core
    ${CUDA_LIBRARIES}
    ${CUDNN_LIBRARY}
)
