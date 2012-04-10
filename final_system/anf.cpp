/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include "svm_wrapper.h"

// The point type used
typedef pcl::PointXYZI PointType;

// Information holder for the full point cloud
struct GlobalData
{
  pcl::IndicesPtr indices;
  float x_min, y_min, z_min, i_min, x_size, y_size, z_size, i_size;
  float density, scale;
  int cardinality;
  std::vector<pcl::svmData> features;

  GlobalData () :
    indices (new std::vector<int>),
    x_min (std::numeric_limits<float>::max ()), y_min (std::numeric_limits<float>::max ()),
    z_min (std::numeric_limits<float>::max ()), i_min (std::numeric_limits<float>::max ()),
    x_size (std::numeric_limits<float>::min ()), y_size (std::numeric_limits<float>::min ()),
    z_size (std::numeric_limits<float>::min ()), i_size (std::numeric_limits<float>::min ()),
    density (), scale (), cardinality (), features ()
  {}
};

// Information holder for each cluster
struct ClusterData
{
  pcl::IndicesPtr indices;
  //std::vector<float> features;
  bool is_isolated, is_tree, is_ghost;

  ClusterData () :
    indices (new std::vector<int>),
    //features (),
    is_isolated (false), is_tree (false), is_ghost (false)
  {}
};

// The computation pipeline stages
#include "src/global_information.cpp"
#include "src/plane_segmentation.cpp"
#include "src/object_clustering.cpp"
#include "src/cluster_information.cpp"
#include "src/object_classification.cpp"
#include "src/noise_filtering.cpp"

void
compute (const pcl::PointCloud<PointType>::Ptr cloud_in, pcl::PointCloud<PointType>::Ptr &cloud_out, float scale, const char *model=NULL)
{
  pcl::console::TicToc tt;
  pcl::console::print_highlight (stderr, "Computing (1/6): Global information ");
  tt.tic ();

  GlobalData global_data;
  gatherGlobalInformation (cloud_in, global_data);
  global_data.scale = scale;

  pcl::console::print_info ("[done, ");
  pcl::console::print_value ("%g", tt.toc ());
  pcl::console::print_info (" ms]\n");
  pcl::console::print_info (stderr, "Finite points in cloud: ");
  pcl::console::print_value (stderr, "%d\n", global_data.indices->size ());
  pcl::console::print_highlight (stderr, "Computing (2/6): Plane segmentation ");
  tt.tic ();

  applyPlaneSegmentation (cloud_in, global_data);

  pcl::console::print_info ("[done, ");
  pcl::console::print_value ("%g", tt.toc ());
  pcl::console::print_info (" ms]\n");
  pcl::console::print_info (stderr, "Remaining points to work on: ");
  pcl::console::print_value (stderr, "%d\n", global_data.indices->size ());
  pcl::console::print_highlight (stderr, "Computing (3/6): Object clustering ");
  tt.tic ();

  boost::shared_ptr<std::vector<ClusterData> > clusters_data (new std::vector<ClusterData>);
  applyObjectClustering (cloud_in, global_data, clusters_data);

  pcl::console::print_info ("[done, ");
  pcl::console::print_value ("%g", tt.toc ());
  pcl::console::print_info (" ms]\n");
  pcl::console::print_info (stderr, "Number of clusters found: ");
  pcl::console::print_value (stderr, "%d\n", clusters_data->size ());
  pcl::console::print_highlight (stderr, "Computing (4/6): Cluster information ");
  tt.tic ();

  gatherClusterInformation (cloud_in, clusters_data, global_data);

  pcl::console::print_info ("[done, ");
  pcl::console::print_value ("%g", tt.toc ());
  pcl::console::print_info (" ms]\n");
  pcl::console::print_highlight (stderr, "Computing (5/6): Object classification ");
  tt.tic ();

  applyObjectClassification (cloud_in, global_data, clusters_data, model);

  pcl::console::print_info ("[done, ");
  pcl::console::print_value ("%g", tt.toc ());
  pcl::console::print_info (" ms]\n");
  pcl::console::print_highlight (stderr, "Computing (6/6): Noise filtering ");
  tt.tic ();

  applyNoiseFiltering (cloud_in, global_data, clusters_data, cloud_out);

  pcl::console::print_info ("[done, ");
  pcl::console::print_value ("%g", tt.toc ());
  pcl::console::print_info (" ms]\n");
}

void
saveCloud (const std::string filename, const pcl::PointCloud<PointType>::Ptr cloud)
{
  pcl::console::TicToc tt;
  pcl::console::print_highlight (stderr, "Saving ");
  pcl::console::print_value (stderr, "%s ", filename.c_str ());
  tt.tic ();
  pcl::io::savePCDFileBinary (filename, *cloud);
  pcl::console::print_info ("[done, ");
  pcl::console::print_value ("%g", tt.toc ());
  pcl::console::print_info (" ms : ");
  pcl::console::print_value ("%d", cloud->width * cloud->height);
  pcl::console::print_info (" points]\n");
}

bool
loadCloud (const std::string filename, pcl::PointCloud<PointType>::Ptr &cloud)
{
  pcl::console::TicToc tt;
  pcl::console::print_highlight (stderr, "Loading ");
  pcl::console::print_value (stderr, "%s ", filename.c_str ());
  tt.tic ();
  if (pcl::io::loadPCDFile<PointType> (filename, *cloud) < 0)
    return (false);
  pcl::console::print_info ("[done, ");
  pcl::console::print_value ("%g", tt.toc ());
  pcl::console::print_info (" ms : ");
  pcl::console::print_value ("%d", cloud->width * cloud->height);
  pcl::console::print_info (" points]\n");
  pcl::console::print_info ("Available dimensions: ");
  pcl::console::print_value ("%s\n", getFieldsList (*cloud).c_str ());
  return (true);
}

void
printHelp (char **argv)
{
  pcl::console::print_error ("Correct syntax: ");
  pcl::console::print_value ("%s input.pcd output.pcd <options>\n", argv[0]);
  pcl::console::print_info ("Options:\n -scale x   x = the distance of one meter\n");
  pcl::console::print_info (" -model filename   filename = the classifier model filename\n");
}

int
main (int argc, char** argv)
{
  pcl::console::print_highlight ("Automated Noise Filtering by Mattia Di Gaetano and Frits Florentinus\n");

  // Parse pcd arguments
  std::vector<int> arg_indices = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (arg_indices.size () != 2)
  {
    printHelp (argv);
    return (-1);
  }

  // Parse other arguments
  float scale = 1500.0;
  std::string model = argv[arg_indices[0]];
  model.append(".model");
  pcl::console::parse_argument (argc, argv, "-scale", scale);
  pcl::console::parse_argument (argc, argv, "-model", model);

  // Load input cloud
  pcl::PointCloud<PointType>::Ptr cloud_in (new pcl::PointCloud<PointType>);
  if (!loadCloud (argv[arg_indices[0]], cloud_in))
    return (-1);

  // Computation
  pcl::PointCloud<PointType>::Ptr cloud_out (new pcl::PointCloud<PointType>);
  compute (cloud_in, cloud_out, scale, model.data());

  // Save output cloud
  saveCloud (argv[arg_indices[1]], cloud_out);
  return (0);
}
