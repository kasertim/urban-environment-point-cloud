# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canoncical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/DATA/TRCS/urban-environment-point-cloud/mattia/clustering_classification

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/DATA/TRCS/urban-environment-point-cloud/mattia/clustering_classification/build

# Include any dependencies generated for this target.
include CMakeFiles/clustering_classification.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/clustering_classification.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/clustering_classification.dir/flags.make

CMakeFiles/clustering_classification.dir/main.cpp.o: CMakeFiles/clustering_classification.dir/flags.make
CMakeFiles/clustering_classification.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/DATA/TRCS/urban-environment-point-cloud/mattia/clustering_classification/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/clustering_classification.dir/main.cpp.o"
	/home/diga/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/clustering_classification.dir/main.cpp.o -c /media/DATA/TRCS/urban-environment-point-cloud/mattia/clustering_classification/main.cpp

CMakeFiles/clustering_classification.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/clustering_classification.dir/main.cpp.i"
	/home/diga/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/DATA/TRCS/urban-environment-point-cloud/mattia/clustering_classification/main.cpp > CMakeFiles/clustering_classification.dir/main.cpp.i

CMakeFiles/clustering_classification.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/clustering_classification.dir/main.cpp.s"
	/home/diga/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/DATA/TRCS/urban-environment-point-cloud/mattia/clustering_classification/main.cpp -o CMakeFiles/clustering_classification.dir/main.cpp.s

CMakeFiles/clustering_classification.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/clustering_classification.dir/main.cpp.o.requires

CMakeFiles/clustering_classification.dir/main.cpp.o.provides: CMakeFiles/clustering_classification.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/clustering_classification.dir/build.make CMakeFiles/clustering_classification.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/clustering_classification.dir/main.cpp.o.provides

CMakeFiles/clustering_classification.dir/main.cpp.o.provides.build: CMakeFiles/clustering_classification.dir/main.cpp.o

CMakeFiles/clustering_classification.dir/svm.cpp.o: CMakeFiles/clustering_classification.dir/flags.make
CMakeFiles/clustering_classification.dir/svm.cpp.o: ../svm.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/DATA/TRCS/urban-environment-point-cloud/mattia/clustering_classification/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/clustering_classification.dir/svm.cpp.o"
	/home/diga/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/clustering_classification.dir/svm.cpp.o -c /media/DATA/TRCS/urban-environment-point-cloud/mattia/clustering_classification/svm.cpp

CMakeFiles/clustering_classification.dir/svm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/clustering_classification.dir/svm.cpp.i"
	/home/diga/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/DATA/TRCS/urban-environment-point-cloud/mattia/clustering_classification/svm.cpp > CMakeFiles/clustering_classification.dir/svm.cpp.i

CMakeFiles/clustering_classification.dir/svm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/clustering_classification.dir/svm.cpp.s"
	/home/diga/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/DATA/TRCS/urban-environment-point-cloud/mattia/clustering_classification/svm.cpp -o CMakeFiles/clustering_classification.dir/svm.cpp.s

CMakeFiles/clustering_classification.dir/svm.cpp.o.requires:
.PHONY : CMakeFiles/clustering_classification.dir/svm.cpp.o.requires

CMakeFiles/clustering_classification.dir/svm.cpp.o.provides: CMakeFiles/clustering_classification.dir/svm.cpp.o.requires
	$(MAKE) -f CMakeFiles/clustering_classification.dir/build.make CMakeFiles/clustering_classification.dir/svm.cpp.o.provides.build
.PHONY : CMakeFiles/clustering_classification.dir/svm.cpp.o.provides

CMakeFiles/clustering_classification.dir/svm.cpp.o.provides.build: CMakeFiles/clustering_classification.dir/svm.cpp.o

# Object files for target clustering_classification
clustering_classification_OBJECTS = \
"CMakeFiles/clustering_classification.dir/main.cpp.o" \
"CMakeFiles/clustering_classification.dir/svm.cpp.o"

# External object files for target clustering_classification
clustering_classification_EXTERNAL_OBJECTS =

clustering_classification: CMakeFiles/clustering_classification.dir/main.cpp.o
clustering_classification: CMakeFiles/clustering_classification.dir/svm.cpp.o
clustering_classification: /usr/lib/libboost_system-mt.so
clustering_classification: /usr/lib/libboost_filesystem-mt.so
clustering_classification: /usr/lib/libboost_thread-mt.so
clustering_classification: /usr/lib/libboost_date_time-mt.so
clustering_classification: /usr/lib/libboost_iostreams-mt.so
clustering_classification: /usr/lib/libpcl_common.so
clustering_classification: /usr/lib/libflann_cpp_s.a
clustering_classification: /usr/lib/libpcl_kdtree.so
clustering_classification: /usr/lib/libpcl_octree.so
clustering_classification: /usr/lib/libpcl_search.so
clustering_classification: /usr/lib/libpcl_features.so
clustering_classification: /usr/lib/libpcl_sample_consensus.so
clustering_classification: /usr/lib/libpcl_filters.so
clustering_classification: /usr/lib/libpcl_keypoints.so
clustering_classification: /usr/lib/libOpenNI.so
clustering_classification: /usr/lib/libpcl_io.so
clustering_classification: /usr/lib/libpcl_segmentation.so
clustering_classification: /usr/lib/libqhull.so
clustering_classification: /usr/lib/libpcl_surface.so
clustering_classification: /usr/lib/libpcl_registration.so
clustering_classification: /usr/lib/libpcl_visualization.so
clustering_classification: /usr/lib/libpcl_tracking.so
clustering_classification: /usr/lib/libpcl_apps.so
clustering_classification: /usr/lib/i386-linux-gnu/libfreetype.so
clustering_classification: /usr/lib/libgl2ps.so
clustering_classification: /usr/lib/i386-linux-gnu/libXt.so
clustering_classification: /usr/lib/libpq.so
clustering_classification: /usr/lib/libmysqlclient.so
clustering_classification: /usr/lib/i386-linux-gnu/libpng.so
clustering_classification: /usr/lib/i386-linux-gnu/libz.so
clustering_classification: /usr/lib/i386-linux-gnu/libjpeg.so
clustering_classification: /usr/lib/i386-linux-gnu/libtiff.so
clustering_classification: /usr/lib/i386-linux-gnu/libexpat.so
clustering_classification: /usr/lib/libavformat.so
clustering_classification: /usr/lib/libavcodec.so
clustering_classification: /usr/lib/libavutil.so
clustering_classification: /usr/lib/libswscale.so
clustering_classification: /usr/lib/i386-linux-gnu/libGL.so
clustering_classification: /usr/lib/openmpi/lib/libmpi.so
clustering_classification: /usr/lib/openmpi/lib/libopen-rte.so
clustering_classification: /usr/lib/openmpi/lib/libopen-pal.so
clustering_classification: /usr/lib/i386-linux-gnu/libdl.so
clustering_classification: /usr/lib/i386-linux-gnu/libnsl.so
clustering_classification: /usr/lib/i386-linux-gnu/libutil.so
clustering_classification: /usr/lib/i386-linux-gnu/libm.so
clustering_classification: /usr/lib/i386-linux-gnu/libdl.so
clustering_classification: /usr/lib/i386-linux-gnu/libnsl.so
clustering_classification: /usr/lib/i386-linux-gnu/libutil.so
clustering_classification: /usr/lib/i386-linux-gnu/libm.so
clustering_classification: /usr/lib/openmpi/lib/libmpi_cxx.so
clustering_classification: CMakeFiles/clustering_classification.dir/build.make
clustering_classification: CMakeFiles/clustering_classification.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable clustering_classification"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/clustering_classification.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/clustering_classification.dir/build: clustering_classification
.PHONY : CMakeFiles/clustering_classification.dir/build

CMakeFiles/clustering_classification.dir/requires: CMakeFiles/clustering_classification.dir/main.cpp.o.requires
CMakeFiles/clustering_classification.dir/requires: CMakeFiles/clustering_classification.dir/svm.cpp.o.requires
.PHONY : CMakeFiles/clustering_classification.dir/requires

CMakeFiles/clustering_classification.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/clustering_classification.dir/cmake_clean.cmake
.PHONY : CMakeFiles/clustering_classification.dir/clean

CMakeFiles/clustering_classification.dir/depend:
	cd /media/DATA/TRCS/urban-environment-point-cloud/mattia/clustering_classification/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/DATA/TRCS/urban-environment-point-cloud/mattia/clustering_classification /media/DATA/TRCS/urban-environment-point-cloud/mattia/clustering_classification /media/DATA/TRCS/urban-environment-point-cloud/mattia/clustering_classification/build /media/DATA/TRCS/urban-environment-point-cloud/mattia/clustering_classification/build /media/DATA/TRCS/urban-environment-point-cloud/mattia/clustering_classification/build/CMakeFiles/clustering_classification.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/clustering_classification.dir/depend

