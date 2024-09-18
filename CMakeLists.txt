cmake_minimum_required(VERSION 3.16)

if(WIN32)
    # set(CMAKE_PREFIX_PATH "C:/Qt/6.7.2/msvc2019_64")
	## example for setting the root path to your Qt installation
elseif(UNIX)
    # set(CMAKE_PREFIX_PATH "/path/to/your/qt")
	# usually doesn't need to set specifically if in the system PATH
endif()

project(ChessAI LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(Qt6 COMPONENTS Widgets Multimedia REQUIRED)

file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "include/*.h")

qt_add_resources(RESOURCES resources.qrc)

add_executable(ChessAI
    ${SOURCES}
    ${HEADERS}
    ${RESOURCES}
)

# Add this line to include the 'include' directory
target_include_directories(ChessAI PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(ChessAI PRIVATE
    Qt6::Widgets
    Qt6::Multimedia
)
