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
include CMakeFiles/benchmark_eval.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/benchmark_eval.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/benchmark_eval.dir/flags.make

CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.o: CMakeFiles/benchmark_eval.dir/flags.make
CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.o: ../benchmark-compare.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/diga/projects/benchmark/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.o"
	/home/diga/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.o -c /home/diga/projects/benchmark/benchmark-compare.cpp

CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.i"
	/home/diga/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/diga/projects/benchmark/benchmark-compare.cpp > CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.i

CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.s"
	/home/diga/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/diga/projects/benchmark/benchmark-compare.cpp -o CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.s

CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.o.requires:
.PHONY : CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.o.requires

CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.o.provides: CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.o.requires
	$(MAKE) -f CMakeFiles/benchmark_eval.dir/build.make CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.o.provides.build
.PHONY : CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.o.provides

CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.o.provides.build: CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.o

# Object files for target benchmark_eval
benchmark_eval_OBJECTS = \
"CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.o"

# External object files for target benchmark_eval
benchmark_eval_EXTERNAL_OBJECTS =

benchmark_eval: CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.o
benchmark_eval: /usr/lib/libboost_system-mt.so
benchmark_eval: /usr/lib/libboost_filesystem-mt.so
benchmark_eval: /usr/lib/libboost_thread-mt.so
benchmark_eval: /usr/lib/libboost_date_time-mt.so
benchmark_eval: /usr/lib/libboost_iostreams-mt.so
benchmark_eval: /usr/lib/libpcl_common.so
benchmark_eval: /usr/lib/libflann_cpp_s.a
benchmark_eval: /usr/lib/libpcl_kdtree.so
benchmark_eval: /usr/lib/libpcl_octree.so
benchmark_eval: /usr/lib/libpcl_search.so
benchmark_eval: /usr/lib/libpcl_features.so
benchmark_eval: /usr/lib/libpcl_sample_consensus.so
benchmark_eval: /usr/lib/libpcl_filters.so
benchmark_eval: /usr/lib/libpcl_keypoints.so
benchmark_eval: /usr/lib/libOpenNI.so
benchmark_eval: /usr/lib/libpcl_io.so
benchmark_eval: /usr/lib/libpcl_segmentation.so
benchmark_eval: /usr/lib/libqhull.so
benchmark_eval: /usr/lib/libpcl_surface.so
benchmark_eval: /usr/lib/libpcl_registration.so
benchmark_eval: /usr/lib/libpcl_visualization.so
benchmark_eval: /usr/lib/libpcl_tracking.so
benchmark_eval: /usr/lib/libpcl_apps.so
benchmark_eval: /usr/lib/i386-linux-gnu/libfreetype.so
benchmark_eval: /usr/lib/libgl2ps.so
benchmark_eval: /usr/lib/i386-linux-gnu/libXt.so
benchmark_eval: /usr/lib/libpq.so
benchmark_eval: /usr/lib/libmysqlclient.so
benchmark_eval: /usr/lib/i386-linux-gnu/libpng.so
benchmark_eval: /usr/lib/i386-linux-gnu/libz.so
benchmark_eval: /usr/lib/i386-linux-gnu/libjpeg.so
benchmark_eval: /usr/lib/i386-linux-gnu/libtiff.so
benchmark_eval: /usr/lib/i386-linux-gnu/libexpat.so
benchmark_eval: /usr/lib/libavformat.so
benchmark_eval: /usr/lib/libavcodec.so
benchmark_eval: /usr/lib/libavutil.so
benchmark_eval: /usr/lib/libswscale.so
benchmark_eval: /usr/lib/i386-linux-gnu/libGL.so
benchmark_eval: /usr/lib/openmpi/lib/libmpi.so
benchmark_eval: /usr/lib/openmpi/lib/libopen-rte.so
benchmark_eval: /usr/lib/openmpi/lib/libopen-pal.so
benchmark_eval: /usr/lib/i386-linux-gnu/libdl.so
benchmark_eval: /usr/lib/i386-linux-gnu/libnsl.so
benchmark_eval: /usr/lib/i386-linux-gnu/libutil.so
benchmark_eval: /usr/lib/i386-linux-gnu/libm.so
benchmark_eval: /usr/lib/i386-linux-gnu/libdl.so
benchmark_eval: /usr/lib/i386-linux-gnu/libnsl.so
benchmark_eval: /usr/lib/i386-linux-gnu/libutil.so
benchmark_eval: /usr/lib/i386-linux-gnu/libm.so
benchmark_eval: /usr/lib/openmpi/lib/libmpi_cxx.so
benchmark_eval: CMakeFiles/benchmark_eval.dir/build.make
benchmark_eval: CMakeFiles/benchmark_eval.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable benchmark_eval"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/benchmark_eval.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/benchmark_eval.dir/build: benchmark_eval
.PHONY : CMakeFiles/benchmark_eval.dir/build

CMakeFiles/benchmark_eval.dir/requires: CMakeFiles/benchmark_eval.dir/benchmark-compare.cpp.o.requires
.PHONY : CMakeFiles/benchmark_eval.dir/requires

CMakeFiles/benchmark_eval.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/benchmark_eval.dir/cmake_clean.cmake
.PHONY : CMakeFiles/benchmark_eval.dir/clean

CMakeFiles/benchmark_eval.dir/depend:
	cd /home/diga/projects/benchmark/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/diga/projects/benchmark /home/diga/projects/benchmark /home/diga/projects/benchmark/build /home/diga/projects/benchmark/build /home/diga/projects/benchmark/build/CMakeFiles/benchmark_eval.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/benchmark_eval.dir/depend

