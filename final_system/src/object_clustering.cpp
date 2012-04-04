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

// TODO: Apply a sub-segmentation in order to separate trunks from leaves for instance or separate adacent objects

/** \brief Divides the remaining points into several clusters, each cluster likely to contain exactly one object.
 * \param[in] cloud_in A pointer to the input point cloud.
 * \param[in] global_data A struct holding information on the full point cloud and global input parameters.
 * \param[out] clusters_data An array of information holders for each cluster
 */
void
applyObjectClustering (const pcl::PointCloud<PointType>::Ptr cloud_in, GlobalData global_data,
                       boost::shared_ptr<std::vector<ClusterData> > &clusters_data)
{
  // Input parameters for this section:
  float resolution = 0.5 * global_data.scale; // Divide the space in blocks of 0.5x0.5x0.5 meter
//  float resolution = 0.0333 * (global_data.x_size + global_data.y_size + global_data.z_size); // Divide the space in approximately 10x10x10 blocks
  int max_cluster_size = 25000; // Do something smart here

  // An octree representation class for downsampling
  pcl::octree::OctreePointCloudSearch<PointType> octree (resolution);
  octree.setInputCloud (cloud_in, global_data.indices);
  octree.addPointsFromInputCloud ();

  // A EuclideanClusterExtraction class for initial clustering (under-segmentation)
  pcl::EuclideanClusterExtraction<PointType> ece;
  ece.setClusterTolerance (1.1 * resolution);
  ece.setMinClusterSize (1);
  ece.setMaxClusterSize (max_cluster_size);

  // Variables used
  pcl::search::KdTree<PointType>::Ptr searcher (new pcl::search::KdTree<PointType>);
  pcl::PointCloud<PointType>::Ptr cloud_octree (new pcl::PointCloud<PointType>);
  std::vector<pcl::PointIndices> clustering;

  // Downsampling
  octree.getOccupiedVoxelCenters (cloud_octree->points);
  cloud_octree->width = cloud_octree->points.size ();
  cloud_octree->height = 1;

  // Under segmentation clustering
  ece.setInputCloud (cloud_octree);
  ece.setSearchMethod (searcher);
  ece.extract (clustering);

//  pcl::io::savePCDFileBinary ("temp.pcd", *cloud_octree);

  // Upsample back from octree representation and store in output
  clusters_data->resize (clustering.size ());
  for (size_t c_it = 0; c_it < clustering.size (); ++c_it)
  {
    (*clusters_data)[c_it].indices = boost::make_shared<std::vector<int> > ();
    for (size_t ci_it = 0; ci_it < clustering[c_it].indices.size (); ++ci_it)
    {
      std::vector<int > voxel_indices;
      octree.voxelSearch (cloud_octree->points[clustering[c_it].indices[ci_it]], voxel_indices);
      (*clusters_data)[c_it].indices->insert ((*clusters_data)[c_it].indices->end (), voxel_indices.begin (), voxel_indices.end ());
    }
  }
}
