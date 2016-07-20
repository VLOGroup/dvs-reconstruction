## This CMake file assumes the following environment variables to be set:
# CUDA_SDK_ROOT_DIR
# MATLAB_ROOT_DIR

PROJECT(LIVE_DENOISE)
cmake_minimum_required(VERSION 2.8)
FILE(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" OT_CMAKE_MODULE_PATH)
SET(CMAKE_MODULE_PATH ${OT_CMAKE_MODULE_PATH})
#SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#SET(CMAKE_BUILD_TYPE Release)
OPTION(WITH_OPENMP "Build with Multiprocessor Support" ON)
if(WITH_OPENMP)
    # for openmp
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    add_definitions("-O3 -DPARALLEL -ffast-math")
    set(CMAKE_BUILD_TYPE Release)
else(WITH_OPENMP)
    add_definitions("-ggdb")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-G -keep")
    set(CMAKE_BUILD_TYPE Debug)
endif(WITH_OPENMP)
##-----------------------------------------------------------------------------
## Compiler Flags
if(WIN32)
  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT.lib /MDd")
endif(WIN32)
add_definitions("-std=c++11 -fpermissive")

##-----------------------------------------------------------------------------
## ImageUtilities
#change the following line to whatever graphics card you have
set(ImageUtilitiesLight_DIR $ENV{IMAGEUTILITIES_ROOT})
set(IMAGEUTILITIESLIGHT_PREFER_STATIC_LIBRARIES true)
find_package(ImageUtilitiesLight REQUIRED COMPONENTS iucore iuio iugui iumath)
cuda_include_directories(${IMAGEUTILITIESLIGHT_INCLUDE_DIR})
include_directories(${IMAGEUTILITIESLIGHT_INCLUDE_DIR})
message(${IMAGEUTILITIESLIGHT_LIBRARIES})

##-----------------------------------------------------------------------------
## OpenCV
#FIND_PACKAGE(OpenCV REQUIRED)
#INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
#CUDA_INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
#message("Found OpenCV Version: " ${OpenCV_VERSION})

##-----------------------------------------------------------------------------
## Additional opencv packages
#find_package( OpenCV REQUIRED COMPONENTS opencv_calib3d)


##-----------------------------------------------------------------------------
## Qt5
set(CMAKE_AUTOMOC ON)
find_package(Qt5Core)
find_package(Qt5Widgets)
find_package(Qt5OpenGL)
qt5_add_resources(UI_RESOURCES ${CMAKE_CURRENT_SOURCE_DIR}/resources/resources.qrc)
#find_package(Qt5Xml)
#include_directories(${Qt5Xml_INCLUDE_DIRS})
#set(LIBQLVIEWER_LIB  /usr/local/lib/libQGLViewer.so)
#find_package( OpenGL )

#find_package(ITK REQUIRED)
#include(${ITK_USE_FILE})

find_package(Boost COMPONENTS program_options filesystem system REQUIRED)

SET(CUDA_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/common.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/denoise.cu )
SET(HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/event.h
  ${CMAKE_CURRENT_SOURCE_DIR}/scopedtimer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/common.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/denoise.h)

if(WIN32)
	cuda_add_library(common  ${CUDA_FILES})
else(WIN32)
  cuda_add_library(common STATIC ${CUDA_FILES})
  target_link_libraries(common ${IMAGEUTILITIESLIGHT_LIBRARIES})
endif(WIN32)
target_link_libraries(common ${OpenCV_LIBRARIES} ${OpenCV_LIBS})

SET ( GUI_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/live_denoise_gui.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/denoisingmainwindow.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/denoisingworker.cpp
  #${CMAKE_CURRENT_SOURCE_DIR}/denoising.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dvscameraworker.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/daviscameraworker.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cnpy.cpp)

SET (TEST_MAIN_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/denoise_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cnpy.cpp
)
link_directories(/usr/local/lib/x86_64-linux-gnu/) # libcaer
#target_link_libraries(common ${OpenCV_LIBRARIES} ${OpenCV_LIBS})
CUDA_ADD_EXECUTABLE(live_reconstruction_gui ${GUI_FILES} ${HEADER_FILES} ${UI_RESOURCES})
TARGET_LINK_LIBRARIES(live_reconstruction_gui common X11 Qt5::Widgets Qt5::OpenGL caer)
CUDA_ADD_EXECUTABLE(denoising_test ${TEST_MAIN_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(denoising_test ${IU_LIBRARIES} ${CUDA_LIBRARIES} ${Boost_LIBRARIES} common)
