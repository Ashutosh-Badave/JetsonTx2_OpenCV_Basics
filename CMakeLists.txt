cmake_minimum_required(VERSION 3.10)
project(opencv_basics)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV 4.3.0 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(jetson_camera Jetson_camera_launch.cpp)
target_link_libraries(jetson_camera ${OpenCV_LIBS})

add_executable(GaussianBlur_and_Gradient GaussianBlur_and_Gradient.cpp BCR_adjustment.hpp)
target_link_libraries(GaussianBlur_and_Gradient ${OpenCV_LIBS})

add_executable(CornerHarris_NHS CornerHarris_and_NHS.cpp BCR_adjustment.hpp)
target_link_libraries(CornerHarris_NHS ${OpenCV_LIBS})

add_executable(detector_FAST detectorFAST.cpp BCR_adjustment.hpp)
target_link_libraries(detector_FAST ${OpenCV_LIBS})