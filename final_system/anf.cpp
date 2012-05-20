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
#include <pcl/octree/octree.h>
#include "svm_wrapper.h"

// The point type used
typedef pcl::PointXYZI PointType;

// Information holder for globals and some point cloud characteristics
struct GlobalData
{
  float x_min, y_min, z_min, i_min, x_size, y_size, z_size, i_size;
  float scale, cagg, pground, plarge, psmall;
  std::string model, cloud_name; // I'd rather see the I/O in anf.cpp and that the contents is passed to SVM
  bool train, octrees;
  pcl::PointCloud<PointType>::Ptr cloud_octree;
  pcl::octree::OctreePointCloudSearch<PointType> octree;

  GlobalData () :
      x_min (std::numeric_limits<float>::max ()), y_min (std::numeric_limits<float>::max ()),
      z_min (std::numeric_limits<float>::max ()), i_min (std::numeric_limits<float>::max ()),
      x_size (std::numeric_limits<float>::min ()), y_size (std::numeric_limits<float>::min ()),
      z_size (std::numeric_limits<float>::min ()), i_size (std::numeric_limits<float>::min ()),
      scale (1500.0), cagg (0.5), pground (0.05), plarge (0.1), psmall (0.001),
      octrees (false),
      octree(0),
      cloud_octree(new pcl::PointCloud<PointType>)
  {}
};

// Information holder for each cluster
struct ClusterData
{
  pcl::IndicesPtr indices; // cluster indices
  pcl::IndicesPtr octree_indices; // cluster indices in the octree representation
  pcl::SVMData features;
  // Classification bool
  bool is_isolated, is_good, is_tree, is_ghost;

  // Classification probabilities
  float is_good_prob, is_tree_prob, is_ghost_prob;
  
  // Subclustering indices & features
  std::vector<pcl::PointIndices> sub_indices;
  std::vector<pcl::SVMData> sub_features;

  ClusterData () :
      indices (new std::vector<int>),
      octree_indices (new std::vector<int>),
      features (),
      is_isolated (false), is_good (false), is_tree (false), is_ghost (false),
      is_good_prob (0.0), is_tree_prob (0.0), is_ghost_prob (0.0)
  {}
};

// A GlobalData instance
GlobalData global_data;

// The computation pipeline stages
#include "src/global_information.cpp"
#include "src/plane_segmentation.cpp"
#include "src/object_clustering.cpp"
#include "src/cluster_information_v2.cpp"
#include "src/object_classification_v3.cpp"
#include "src/noise_filtering.cpp"

void
compute (const pcl::PointCloud<PointType>::Ptr cloud_in, pcl::PointCloud<PointType>::Ptr &cloud_out)
{
  pcl::console::TicToc tt;
  pcl::console::print_highlight (stderr, "Computing (1/6): Global information ");
  tt.tic ();

  pcl::IndicesPtr indices (new std::vector<int>);
  gatherGlobalInformation (cloud_in, indices);

  pcl::console::print_info ("[done, ");
  pcl::console::print_value ("%g", tt.toc ());
  pcl::console::print_info (" ms]\n");
  pcl::console::print_info (stderr, "Finite points in cloud: ");
  pcl::console::print_value (stderr, "%d\n", indices->size ());
  pcl::console::print_highlight (stderr, "Computing (2/6): Plane segmentation ");
  tt.tic ();

  applyPlaneSegmentation (cloud_in, indices);

  pcl::console::print_info ("[done, ");
  pcl::console::print_value ("%g", tt.toc ());
  pcl::console::print_info (" ms]\n");
  pcl::console::print_info (stderr, "Remaining points to work on: ");
  pcl::console::print_value (stderr, "%d\n", indices->size ());
  pcl::console::print_highlight (stderr, "Computing (3/6): Object clustering ");
  tt.tic ();

  boost::shared_ptr<std::vector<ClusterData> > clusters_data (new std::vector<ClusterData>);
  applyObjectClustering (cloud_in, indices, clusters_data);

  pcl::console::print_info ("[done, ");
  pcl::console::print_value ("%g", tt.toc ());
  pcl::console::print_info (" ms]\n");
  pcl::console::print_info (stderr, "Number of clusters found: ");
  pcl::console::print_value (stderr, "%d\n", clusters_data->size ());

  // Moved this here because
  if (global_data.octrees)
    system ("pcl_pcd_viewer -multiview 1 octree_clusters.pcd octree_planes.pcd");

  pcl::console::print_highlight (stderr, "Computing (4/6): Cluster information ");
  tt.tic ();

  gatherClusterInformation (cloud_in, clusters_data);

  pcl::console::print_info ("[done, ");
  pcl::console::print_value ("%g", tt.toc ());
  pcl::console::print_info (" ms]\n");
  pcl::console::print_highlight (stderr, "Computing (5/6): Object classification ");
  tt.tic ();

  applyObjectClassification (cloud_in, clusters_data);

  pcl::console::print_info ("[done, ");
  pcl::console::print_value ("%g", tt.toc ());
  pcl::console::print_info (" ms]\n");
  pcl::console::print_highlight (stderr, "Computing (6/6): Noise filtering ");
  tt.tic ();

  applyNoiseFiltering (cloud_in, clusters_data, cloud_out);

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
  pcl::console::print_value ("%s input.pcd output.pcd svm_classify.model <options>\n", argv[0]);
  pcl::console::print_info ("Options:\n");
  //pcl::console::print_info (" -train     if present, use the input data to train the SVM and append to the model\n");
  pcl::console::print_info (" -scale x   x = distance of one meter\n");
  pcl::console::print_info (" -cagg x    x = aggressiveness of the clustering step (1.0 = prone to over-segment, 0.0 = prone to under-segment, default = 0.5)\n");
  pcl::console::print_info (" -pground x x = the percentage of ground planes\n");
  pcl::console::print_info (" -plarge x  x = the percentage limit for large clusters\n");
  pcl::console::print_info (" -psmall x  x = the percentage limit for isolated clusters\n");
  pcl::console::print_info (" -octrees   if present, output the intermediate octree representations of the segmentation steps, useful for parameter tweaking\n");
}

int
main (int argc, char** argv)
{
  pcl::console::print_highlight ("Automated Noise Filtering by Mattia Di Gaetano and Frits Florentinus\n");

  // Parse pcd arguments
  std::vector<int> pcd_indices = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (pcd_indices.size () != 2)
  {
    printHelp (argv);
    return (-1);
  } 
  else
    global_data.cloud_name.assign(argv[pcd_indices[0]]);

  // Parse model arguments
  std::vector<int> model_indices = pcl::console::parse_file_extension_argument (argc, argv, ".model");
  if (model_indices.size () > 0)
    global_data.model = argv[model_indices[0]];

  // Parse other arguments
  pcl::console::parse_argument (argc, argv, "-scale", global_data.scale);
  pcl::console::parse_argument (argc, argv, "-cagg", global_data.cagg);
  pcl::console::parse_argument (argc, argv, "-pground", global_data.pground);
  pcl::console::parse_argument (argc, argv, "-plarge", global_data.plarge);
  pcl::console::parse_argument (argc, argv, "-psmall", global_data.psmall);
  global_data.octrees = pcl::console::find_switch (argc, argv, "-octrees");

  // Load input cloud
  pcl::PointCloud<PointType>::Ptr cloud_in (new pcl::PointCloud<PointType>);
  if (!loadCloud (argv[pcd_indices[0]], cloud_in))
    return (-1);

  // Computation
  pcl::PointCloud<PointType>::Ptr cloud_out (new pcl::PointCloud<PointType>);
  compute (cloud_in, cloud_out);

  // Save output cloud
  saveCloud (argv[pcd_indices[1]], cloud_out);
  return (0);
}
