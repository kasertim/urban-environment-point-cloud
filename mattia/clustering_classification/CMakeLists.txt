project(clustering_classification)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
find_package(PCL 1.6 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(clustering_classification main.cpp svm.cpp  svm-bus.cpp)
TARGET_LINK_LIBRARIES( clustering_classification ${PCL_LIBRARIES})


project(test_predictor)
add_executable(test_predictor test_predictor.cpp svm.cpp svm-bus.cpp)
TARGET_LINK_LIBRARIES(test_predictor ${PCL_LIBRARIES})
