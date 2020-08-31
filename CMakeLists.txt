cmake_minimum_required(VERSION 3.10)
project(opencv_basics)

set(CMAKE_CXX_STANDARD 11)
find_package( OpenCV 4.3.0 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(opencv_basics Jetson_camera_launch.cpp)

target_link_libraries(opencv_basics ${OpenCV_LIBS})