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

// TODO: Make the cluster_size parameters adaptive
// TODO: Apply the sub-segmentation in order to separate trunks from leaves for instance or separate adjacent objects

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
  // ---- AUTOMATED INPUT PARAMETERS OF THIS SECTION ----
  // Speed / accuracy tradeoff for the first phase of this section
  // Decreasing this number increases the detail of the octree representation and affects the resulting cluster sizes
  // Setting this too small could result in undesirable over-segmentation and also reduces speed too much
  // Setting this too high could result in too much under-segmentation, rendering the first phase almost useless
  // Currently: analyze on voxels of 0.25 x 0.25 x 0.25 meter
  float resolution = 0.5 * global_data.scale;
  // Clusters within this distance (octree representation) from one another are considered the same cluster
  // This value should be slightly larger than sqrt(1), sqrt(2) or sqrt(3) depending on how you want to deal with diagonals
  // Currently: Slightly larger than sqrt(1)
  float distance_threshold = 1.1 * resolution;
  // Beyond this size (octree representation) are not to be classified by the SVM
  // Do something smart here, based on resolution and density
  int max_cluster_size = 500;
  // Below this size (octree representation) are classified as isolated points
  // Do something smart here, based on resolution and density
  int min_cluster_size = 4;

  // ---- PHASE ONE : UNDER SEGMENTATION AND SIZE PASSTHROUGH ----

  // An octree representation class for temporary downsampling
  pcl::octree::OctreePointCloudSearch<PointType> octree (resolution);
  octree.setInputCloud (cloud_in, global_data.indices);
  octree.addPointsFromInputCloud ();

  // A EuclideanClusterExtraction class for clustering
  pcl::EuclideanClusterExtraction<PointType> ece;
  ece.setClusterTolerance (distance_threshold);
  ece.setMinClusterSize (1);
  ece.setMaxClusterSize (max_cluster_size);

  // Variables used
  pcl::PointCloud<PointType>::Ptr cloud_octree (new pcl::PointCloud<PointType>);
  pcl::search::KdTree<PointType>::Ptr searcher (new pcl::search::KdTree<PointType>);
  std::vector<pcl::PointIndices> clustering;

  // Downsampling
  octree.getOccupiedVoxelCenters (cloud_octree->points);
  cloud_octree->width = cloud_octree->points.size ();
  cloud_octree->height = 1;

  // Clustering
  ece.setInputCloud (cloud_octree);
  ece.setSearchMethod (searcher);
  ece.extract (clustering);

  // Upsample back from octree representation and store in output
  clusters_data->resize (clustering.size ());
  for (size_t c_it = 0; c_it < clustering.size (); ++c_it)
  {
    (*clusters_data)[c_it].indices = boost::make_shared<std::vector<int> > ();
    for (size_t ci_it = 0; ci_it < clustering[c_it].indices.size (); ++ci_it)
    {
      std::vector<int> voxel_indices;
      octree.voxelSearch (cloud_octree->points[clustering[c_it].indices[ci_it]], voxel_indices);
      (*clusters_data)[c_it].indices->insert ((*clusters_data)[c_it].indices->end (), voxel_indices.begin (), voxel_indices.end ());
    }
    if (clustering[c_it].indices.size () < min_cluster_size)
      (*clusters_data)[c_it].is_isolated = true;
  }

  // ---- PHASE TWO : ZOOMED SEGMENTATION ----

//  for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
//    if (!(*clusters_data)[c_it].is_isolated)
//      for (size_t ci_it = 0; ci_it < (*clusters_data)[c_it].indices->size (); ++ci_it)
//        (*cloud_in)[(*(*clusters_data)[c_it].indices)[ci_it]].intensity = 100 * c_it;
//  pcl::io::savePCDFileBinary ("temp.pcd", *cloud_in);
//
//  pcl::PointCloud<PointType>::Ptr temp (new pcl::PointCloud<PointType>);
//  pcl::IndicesPtr noise_indices (new std::vector<int>);
//  for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
//    if (!(*clusters_data)[c_it].is_isolated)
//      noise_indices->insert (noise_indices->end (), (*clusters_data)[c_it].indices->begin (), (*clusters_data)[c_it].indices->end ());
//  pcl::ExtractIndices<PointType> ei;
//  ei.setInputCloud (cloud_in);
//  ei.setIndices (noise_indices);
//  ei.setKeepOrganized (true);
//  ei.filter (*temp);
//  for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
//    for (size_t ci_it = 0; ci_it < (*clusters_data)[c_it].indices->size (); ++ci_it)
//      (*temp)[(*(*clusters_data)[c_it].indices)[ci_it]].intensity = c_it;
//  pcl::io::savePCDFileBinary ("temp.pcd", *temp);
}
