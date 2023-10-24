cmake_minimum_required(VERSION 3.22.1)
project(AutoSegmentLulcCRF VERSION 1.0)
set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

find_package(OpenCV 4 REQUIRED core features2d highgui imgproc imgcodecs ml PATHS "$ENV{OPENCVDIR}/build")
find_package(DGM REQUIRED)
find_package(PROJ REQUIRED CONFIG)
find_package(GDAL REQUIRED CONFIG PATHS /usr/local/include)

message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "DGM_INCLUDE_DIRS = ${DGM_INCLUDE_DIRS}")
message(STATUS "PROJ_INCLUDE_DIRS = ${PROJ_INCLUDE_DIRS}")


add_executable(densecrf_inference densecrf_inference.cc)
target_link_libraries(densecrf_inference 
                        PRIVATE "${OpenCV_LIBS}"
                        PRIVATE PROJ::proj 
                        PRIVATE GDAL::GDAL)