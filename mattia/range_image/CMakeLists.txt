project(range_image)

find_package(PCL 1.5 REQUIRED)
FIND_PACKAGE( OpenCV REQUIRED )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(range_image range.cpp)
target_link_libraries (range_image ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(crop_cloud crop.cpp)
target_link_libraries (crop_cloud ${PCL_LIBRARIES})