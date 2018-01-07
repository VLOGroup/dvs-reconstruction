PROJECT(LIVE_DVS_RECONSTRUCTION)
cmake_minimum_required(VERSION 2.8)
FILE(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" OT_CMAKE_MODULE_PATH)
SET(CMAKE_MODULE_PATH ${OT_CMAKE_MODULE_PATH})

OPTION(WITH_OPENMP "Build with Multiprocessor Support" ON)
OPTION(WITH_DAVIS "Build with support for DAVIS240 instead of DVS128" OFF)
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
## ImageUtilities
#change the following line to whatever graphics card you have
set(ImageUtilities_DIR $ENV{IMAGEUTILITIES_ROOT})
set(IMAGEUTILITIES_PREFER_STATIC_LIBRARIES true)
find_package(ImageUtilities REQUIRED COMPONENTS iucore iuio iugui iumath)
cuda_include_directories(${IMAGEUTILITIE_INCLUDE_DIR})
include_directories(${IMAGEUTILITIES_INCLUDE_DIR})
message(${IMAGEUTILITIES_LIBRARIES})

## Boost
find_package(Boost COMPONENTS program_options filesystem system REQUIRED)


##-----------------------------------------------------------------------------
## Qt5
set(CMAKE_AUTOMOC ON)
find_package(Qt5Core)
find_package(Qt5Widgets)
find_package(Qt5OpenGL)
qt5_add_resources(UI_RESOURCES ${CMAKE_CURRENT_SOURCE_DIR}/resources/resources.qrc)

SET(CUDA_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/common.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/denoise.cu )
SET(HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/event.h
  ${CMAKE_CURRENT_SOURCE_DIR}/common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/common.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/denoise.h)

if(WIN32)
	cuda_add_library(common  ${CUDA_FILES})
else(WIN32)
  cuda_add_library(common STATIC ${CUDA_FILES})
  target_link_libraries(common ${IMAGEUTILITIES_LIBRARIES})
endif(WIN32)
target_link_libraries(common ${OpenCV_LIBRARIES} ${OpenCV_LIBS})

SET ( GUI_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/live_reconstruction_gui.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reconstructionmainwindow.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/denoisingworker.cpp)
if(WITH_DAVIS)
  set (GUI_FILES
       ${GUI_FILES}
       ${CMAKE_CURRENT_SOURCE_DIR}/daviscameraworker.cpp)
else(WITH_DAVIS)
   set (GUI_FILES
        ${GUI_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/dvscameraworker.cpp)
endif(WITH_DAVIS)

##-----------------------------------------------------------------------------
## Compiler Flags
if(WIN32)
  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT.lib /MDd")
endif(WIN32)
add_definitions("-std=c++11 -fpermissive")
if(WITH_DAVIS)
  add_definitions("-DDAVIS240")
endif(WITH_DAVIS)

SET (TEST_MAIN_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/offline_reconstruction.cpp
)


link_directories(/usr/local/lib/x86_64-linux-gnu/) # libcaer
#target_link_libraries(common ${OpenCV_LIBRARIES} ${OpenCV_LIBS})
CUDA_ADD_EXECUTABLE(live_reconstruction_gui ${GUI_FILES} ${HEADER_FILES} ${UI_RESOURCES})
TARGET_LINK_LIBRARIES(live_reconstruction_gui common X11 Qt5::Widgets Qt5::OpenGL caer cnpy)
CUDA_ADD_EXECUTABLE(offline_reconstruction ${TEST_MAIN_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(offline_reconstruction ${CUDA_LIBRARIES} ${Boost_LIBRARIES} common cnpy)
