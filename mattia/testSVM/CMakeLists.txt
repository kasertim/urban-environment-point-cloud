project(testsvm)

add_executable(testsvm main.cpp svm.cpp)

project(original)

add_executable(original svm-predict.c svm.cpp)
