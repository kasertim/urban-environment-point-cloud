project(pca_filter)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
find_package(PCL 1.6 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pca_filter main.cpp edge_contour_extraction.hpp)

TARGET_LINK_LIBRARIES(pca_filter ${PCL_LIBRARIES})

add_executable(createcloud createcloud.cpp)
TARGET_LINK_LIBRARIES(createcloud ${PCL_LIBRARIES})
