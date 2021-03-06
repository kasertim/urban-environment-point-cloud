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
CMAKE_SOURCE_DIR = /home/diga/projects/benchmark

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/diga/projects/benchmark/build

# Include any dependencies generated for this target.
include CMakeFiles/benchmark.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/benchmark.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/benchmark.dir/flags.make

CMakeFiles/benchmark.dir/main.cpp.o: CMakeFiles/benchmark.dir/flags.make
CMakeFiles/benchmark.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/diga/projects/benchmark/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/benchmark.dir/main.cpp.o"
	/home/diga/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/benchmark.dir/main.cpp.o -c /home/diga/projects/benchmark/main.cpp

CMakeFiles/benchmark.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmark.dir/main.cpp.i"
	/home/diga/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/diga/projects/benchmark/main.cpp > CMakeFiles/benchmark.dir/main.cpp.i

CMakeFiles/benchmark.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmark.dir/main.cpp.s"
	/home/diga/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/diga/projects/benchmark/main.cpp -o CMakeFiles/benchmark.dir/main.cpp.s

CMakeFiles/benchmark.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/benchmark.dir/main.cpp.o.requires

CMakeFiles/benchmark.dir/main.cpp.o.provides: CMakeFiles/benchmark.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/benchmark.dir/build.make CMakeFiles/benchmark.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/benchmark.dir/main.cpp.o.provides

CMakeFiles/benchmark.dir/main.cpp.o.provides.build: CMakeFiles/benchmark.dir/main.cpp.o

# Object files for target benchmark
benchmark_OBJECTS = \
"CMakeFiles/benchmark.dir/main.cpp.o"

# External object files for target benchmark
benchmark_EXTERNAL_OBJECTS =

benchmark: CMakeFiles/benchmark.dir/main.cpp.o
benchmark: /usr/lib/libboost_system-mt.so
benchmark: /usr/lib/libboost_filesystem-mt.so
benchmark: /usr/lib/libboost_thread-mt.so
benchmark: /usr/lib/libboost_date_time-mt.so
benchmark: /usr/lib/libboost_iostreams-mt.so
benchmark: /usr/lib/libpcl_common.so
benchmark: /usr/lib/libflann_cpp_s.a
benchmark: /usr/lib/libpcl_kdtree.so
benchmark: /usr/lib/libpcl_octree.so
benchmark: /usr/lib/libpcl_search.so
benchmark: /usr/lib/libpcl_features.so
benchmark: /usr/lib/libpcl_sample_consensus.so
benchmark: /usr/lib/libpcl_filters.so
benchmark: /usr/lib/libpcl_keypoints.so
benchmark: /usr/lib/libOpenNI.so
benchmark: /usr/lib/libpcl_io.so
benchmark: /usr/lib/libpcl_segmentation.so
benchmark: /usr/lib/libqhull.so
benchmark: /usr/lib/libpcl_surface.so
benchmark: /usr/lib/libpcl_registration.so
benchmark: /usr/lib/libpcl_visualization.so
benchmark: /usr/lib/libpcl_tracking.so
benchmark: /usr/lib/libpcl_apps.so
benchmark: /usr/lib/i386-linux-gnu/libfreetype.so
benchmark: /usr/lib/libgl2ps.so
benchmark: /usr/lib/i386-linux-gnu/libXt.so
benchmark: /usr/lib/libpq.so
benchmark: /usr/lib/libmysqlclient.so
benchmark: /usr/lib/i386-linux-gnu/libpng.so
benchmark: /usr/lib/i386-linux-gnu/libz.so
benchmark: /usr/lib/i386-linux-gnu/libjpeg.so
benchmark: /usr/lib/i386-linux-gnu/libtiff.so
benchmark: /usr/lib/i386-linux-gnu/libexpat.so
benchmark: /usr/lib/libavformat.so
benchmark: /usr/lib/libavcodec.so
benchmark: /usr/lib/libavutil.so
benchmark: /usr/lib/libswscale.so
benchmark: /usr/lib/i386-linux-gnu/libGL.so
benchmark: /usr/lib/openmpi/lib/libmpi.so
benchmark: /usr/lib/openmpi/lib/libopen-rte.so
benchmark: /usr/lib/openmpi/lib/libopen-pal.so
benchmark: /usr/lib/i386-linux-gnu/libdl.so
benchmark: /usr/lib/i386-linux-gnu/libnsl.so
benchmark: /usr/lib/i386-linux-gnu/libutil.so
benchmark: /usr/lib/i386-linux-gnu/libm.so
benchmark: /usr/lib/i386-linux-gnu/libdl.so
benchmark: /usr/lib/i386-linux-gnu/libnsl.so
benchmark: /usr/lib/i386-linux-gnu/libutil.so
benchmark: /usr/lib/i386-linux-gnu/libm.so
benchmark: /usr/lib/openmpi/lib/libmpi_cxx.so
benchmark: CMakeFiles/benchmark.dir/build.make
benchmark: CMakeFiles/benchmark.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable benchmark"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/benchmark.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/benchmark.dir/build: benchmark
.PHONY : CMakeFiles/benchmark.dir/build

CMakeFiles/benchmark.dir/requires: CMakeFiles/benchmark.dir/main.cpp.o.requires
.PHONY : CMakeFiles/benchmark.dir/requires

CMakeFiles/benchmark.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/benchmark.dir/cmake_clean.cmake
.PHONY : CMakeFiles/benchmark.dir/clean

CMakeFiles/benchmark.dir/depend:
	cd /home/diga/projects/benchmark/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/diga/projects/benchmark /home/diga/projects/benchmark /home/diga/projects/benchmark/build /home/diga/projects/benchmark/build /home/diga/projects/benchmark/build/CMakeFiles/benchmark.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/benchmark.dir/depend

