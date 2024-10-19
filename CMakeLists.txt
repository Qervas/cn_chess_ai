cmake_minimum_required(VERSION 3.16)

if(WIN32)
    # set(CMAKE_PREFIX_PATH "C:/Qt/6.7.2/msvc2019_64")
	## example for setting the root path to your Qt installation
elseif(UNIX)
    # set(CMAKE_PREFIX_PATH "/path/to/your/qt")
	# usually doesn't need to set specifically if in the system PATH
endif()

project(ChessAI LANGUAGES CXX CUDA)


find_package(CUDAToolkit REQUIRED)
find_library(CUDNN_LIBRARY cudnn
  HINTS ${CUDAToolkit_LIBRARY_DIR}
  REQUIRED)

if(NOT CUDNN_LIBRARY)
  message(FATAL_ERROR "cuDNN library not found")
else()
  message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
endif()

# Find cuDNN include directory
find_path(CUDNN_INCLUDE_DIR cudnn.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS}
  REQUIRED)

if(NOT CUDNN_INCLUDE_DIR)
  message(FATAL_ERROR "cuDNN include directory not found")
else()
  message(STATUS "Found cuDNN include: ${CUDNN_INCLUDE_DIR}")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(Qt6 COMPONENTS Widgets Multimedia REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86")

file(GLOB SOURCES "src/*.cpp" "src/*.cu")
file(GLOB HEADERS "include/*.h" "include/*.cuh")

qt6_add_resources(RESOURCES resources.qrc)

add_executable(ChessAI
    ${SOURCES}
    ${HEADERS}
    ${RESOURCES}
)

set_target_properties(ChessAI PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# Add this line to include the 'include' directory
target_include_directories(ChessAI PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDNN_INCLUDE_DIR}
)

target_link_libraries(ChessAI PRIVATE
    Qt6::Widgets
    Qt6::Multimedia
    ${CUDA_LIBRARIES}
    ${CUDNN_LIBRARIES}
)

