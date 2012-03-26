project(clustering_classification)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
find_package(PCL 1.5 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(clustering_classification main.cpp svm.cpp)
TARGET_LINK_LIBRARIES( clustering_classification ${PCL_LIBRARIES})

