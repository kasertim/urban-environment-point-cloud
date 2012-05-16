cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(AutomatedNoiseFiltering)
find_package(PCL 1.6 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
set(LIBS ${LIBS} ${PCL_LIBRARIES})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_definitions(-Wno-deprecated -DEIGEN_DONT_VECTORIZE -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)

SET( SOURCE anf.cpp src/svm/svm.cpp src/svm/svm_wrapper.cpp src/extract_clusters.hpp)

add_executable(anf ${SOURCE})
target_link_libraries(anf ${PCL_LIBRARIES})



