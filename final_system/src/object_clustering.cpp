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

#include <pcl/octree/octree.h>
#include <pcl/segmentation/extract_clusters.h>

/** \brief Divides the remaining points into several clusters, each cluster likely to contain exactly one object.
  * \param[in] cloud_in A pointer to the input point cloud.
  * \param[in] global_data A struct holding information on the full point cloud and global input parameters.
  * \param[out] clusters_data An array of information holders for each cluster
  */
void
applyObjectClustering (const pcl::PointCloud<PointType>::Ptr cloud_in,
                       GlobalData global_data,
                       boost::shared_ptr<std::vector<ClusterData> > &clusters_data)
{
//  // Input parameters for this section:
//  float resolution = global_data.scale; // Divide the space in blocks of 1x1x1 meter
////  float resolution = 0.0333 * (global_data.x_size + global_data.y_size + global_data.z_size); // Divide the space in approximately 10x10x10 blocks
//
//  // An octree representation class
//  pcl::octree::OctreePointCloudSearch<PointType> octree (resolution);
//  octree.setInputCloud (cloud_in, global_data.indices);
//  octree.addPointsFromInputCloud ();
//
//  pcl::PointCloud<PointType>::Ptr octree_cloud (new pcl::PointCloud<PointType>);
//  std::vector<PointType, Eigen::aligned_allocator<PointType> > octree_array;
//  octree.getOccupiedVoxelCenters (octree_array);
//
//  octree_cloud->width = octree_array.size ();
//  octree_cloud->height = 1;
//  octree_cloud->points = octree_array;
//
//  std::vector<pcl::PointIndices> cluster_indices;
//
//  pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
//
//  pcl::EuclideanClusterExtraction < PointType > ec;
//  ec.setClusterTolerance (1.1 * resolution);
//  ec.setMinClusterSize (1);
//  ec.setMaxClusterSize (250000);
//  ec.setSearchMethod (tree);
//  ec.setInputCloud (octree_cloud);
//  ec.extract (cluster_indices);
//
//  for (int i = 0; i < cluster_indices.size (); ++i)
//    for (int j = 0; j < cluster_indices[i].indices.size (); ++j)
//    {
//      std::vector<int > pointIdx_data;
//      octree.voxelSearch (cloud_out->points[cluster_indices[i].indices[j]], pointIdx_data);
//      for (int k = 0; k < static_cast<int> (pointIdx_data.size ()); ++k)
//        cloud_in->points[pointIdx_data[k]].intensity = i;
//    }
//
//  std::cout << "depth: " << octree.getTreeDepth () << std::endl;
//  std::cout << "clusters: " << cluster_indices.size () << std::endl;

  // Passthrough for testing purposes
  clusters_data->resize (1);
  (*clusters_data)[0].indices = global_data.indices;
  (*clusters_data)[0].is_ghost = true;
}
