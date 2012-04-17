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
#include <pcl/features/normal_3d.h>

// TODO: Make the cluster_size parameters adaptive
// TODO: Apply the sub-segmentation in order to separate trunks from leaves for instance or separate adjacent objects

/** \brief Divides the remaining points into several clusters, each cluster likely to contain exactly one object.
 * \param[in] cloud_in A pointer to the input point cloud.
 * \param[in] global_data A struct holding information on the full point cloud and global input parameters.
 * \param[out] clusters_data An array of information holders for each cluster
 */
void
applyObjectClustering (const pcl::PointCloud<PointType>::Ptr cloud_in, GlobalData global_data,
                       boost::shared_ptr<std::vector<ClusterData> > &clusters_data)
{
  // ---- AUTOMATED INPUT PARAMETERS OF THIS SECTION ----
  // Primary speed / accuracy tradeoff for this section.
  // Is also the octree resolution.
  // Setting this too small could result in over-segmentation and also reduces speed too much
  // Setting this too high could result in under-segmentation
  // Currently: analyze on voxels of 0.08 x 0.08 x 0.08 meter with slight alteration based on cluster aggressiveness
  float resolution = 0.08 * global_data.scale / pow (0.5 + global_data.cagg, 2);
  // Clusters within this distance (octree representation) from one another are very likely considered the same cluster.
  // Currently: adjacent points in the octree are considered connected.
  // This value could be slightly larger than sqrt(1), sqrt(2) or sqrt(3) depending on how you want to deal with diagonals.
  float close_distance_threshold = 1.42 * resolution;
  // Clusters within this distance (octree representation) from one another are less likely considered the same cluster.
  // Is also the radius of candidate searching and normal estimation.
  // Currently: a 50 centimeter radius with slight alteration based on cluster aggressiveness
  float far_distance_threshold = 0.50 * global_data.scale / pow (0.5 + global_data.cagg, 2);
  // Close distance points need to satisfy either of these thresholds to pass the similarity check.
  float close_curvature_threshold = 0.06 / pow (0.5 + global_data.cagg, 8);
  float close_intensity_threshold = 0.05 * global_data.i_size / pow (0.5 + global_data.cagg, 8);
  // Far distance points need to satisfy both of these thresholds to pass the similarity check.
  float far_curvature_threshold = 0.006 / pow (0.5 + global_data.cagg, 4);
  float far_intensity_threshold = 0.005 * global_data.i_size / pow (0.5 + global_data.cagg, 4);
  // Below this size (octree representation) are classified as isolated points
  int min_cluster_size = 4;

  // ---- PHASE ONE : UNDER SEGMENTATION AND SIZE PASSTHROUGH ----

  // An octree representation class for temporary downsampling and density normalization
  pcl::octree::OctreePointCloudSearch<PointType> octree (resolution);
  pcl::PointCloud<PointType>::Ptr cloud_octree (new pcl::PointCloud<PointType>);
  octree.setInputCloud (cloud_in, global_data.indices);
  octree.addPointsFromInputCloud ();
  octree.getOccupiedVoxelCenters (cloud_octree->points);
  cloud_octree->width = cloud_octree->points.size ();
  cloud_octree->height = 1;

  // A kdtree class for nearest neighbor searching
  pcl::search::KdTree<PointType>::Ptr searcher (new pcl::search::KdTree<PointType>);
  searcher->setInputCloud (cloud_octree);

  // Feature estimation: Curvature
  pcl::NormalEstimation<PointType, pcl::Normal> ne;
  pcl::PointCloud<pcl::Normal> cloud_normals;
  ne.setInputCloud (cloud_octree);
  ne.setSearchMethod (searcher);
  ne.setRadiusSearch (far_distance_threshold);
  ne.compute (cloud_normals);

  // Feature estimation: Intensity
  std::vector<int> intensity (cloud_octree->width, 0.0);
  for (size_t p_it = 0; p_it < intensity.size (); ++p_it)
  {
    std::vector<int> voxel_indices;
    octree.voxelSearch (cloud_octree->points[p_it], voxel_indices);
    for (size_t vi_it = 0; vi_it < voxel_indices.size (); ++vi_it)
      intensity[p_it] += (*cloud_in).points[voxel_indices[vi_it]].intensity;
    intensity[p_it] /= voxel_indices.size ();
  }

  // Whether points have been assigned a cluster yet
  std::vector<bool> processed (cloud_octree->width, false);

  // Iterate through all points
  for (size_t p_it = 0; p_it < cloud_octree->width; ++p_it)
  {
    if (processed[p_it])
      continue;
    processed[p_it] = true;

    // Region growing from seed point
    std::vector<int> seed_queue;
    seed_queue.push_back (p_it);
    size_t sq_it = 0;
    while (sq_it < seed_queue.size ())
    {
      // Search for sq_it
      std::vector<int> nn_indices;
      std::vector<float> nn_distances;
      if (searcher->radiusSearch (seed_queue[sq_it], far_distance_threshold, nn_indices, nn_distances) < 1)
      {
        sq_it++;
        continue;
      }
      for (size_t nn_it = 1; nn_it < nn_indices.size (); ++nn_it) // nn_indices[0] should be sq_it
      {
        if (processed[nn_indices[nn_it]]) // Has this point been processed before ?
          continue;
        // Does it satisfy all the feature conditions?
        if ((nn_distances[nn_it] < pow (close_distance_threshold, 2)
            && (fabs (cloud_normals.points[nn_indices[0]].curvature - cloud_normals.points[nn_indices[nn_it]].curvature) < close_curvature_threshold
                || fabs (intensity[nn_indices[0]] - intensity[nn_indices[nn_it]]) < close_intensity_threshold))
            || (fabs (cloud_normals.points[nn_indices[0]].curvature - cloud_normals.points[nn_indices[nn_it]].curvature) < far_curvature_threshold
                && fabs (intensity[nn_indices[0]] - intensity[nn_indices[nn_it]]) < far_intensity_threshold))
        {
          seed_queue.push_back (nn_indices[nn_it]);
          processed[nn_indices[nn_it]] = true;
        }
      }
      sq_it++;
    }

    // Upsample back from octree representation and store in output
    ClusterData cluster;
    float color = 512.0 + 2550.0 * rand () / (RAND_MAX + 1.0f);
    for (size_t sq_it = 0; sq_it < seed_queue.size (); ++sq_it)
    {
      cloud_octree->points[seed_queue[sq_it]].intensity = color;
      std::vector<int> voxel_indices;
      octree.voxelSearch (cloud_octree->points[seed_queue[sq_it]], voxel_indices);
      cluster.indices->insert (cluster.indices->end (), voxel_indices.begin (), voxel_indices.end ());
    }
    if (seed_queue.size () < min_cluster_size)
      cluster.is_isolated = true;
    clusters_data->push_back (cluster);
  }

//  // Upsample back from octree representation and store in output
//  clusters_data->resize (clustering.size ());
//  for (size_t c_it = 0; c_it < clustering.size (); ++c_it)
//  {
//    (*clusters_data)[c_it].indices = boost::make_shared<std::vector<int> > ();
//    for (size_t ci_it = 0; ci_it < clustering[c_it].indices.size (); ++ci_it)
//    {
//      std::vector<int> voxel_indices;
//      octree.voxelSearch (cloud_octree->points[clustering[c_it].indices[ci_it]], voxel_indices);
//      (*clusters_data)[c_it].indices->insert ((*clusters_data)[c_it].indices->end (), voxel_indices.begin (), voxel_indices.end ());
//    }
//    if (clustering[c_it].indices.size () < min_cluster_size)
//      (*clusters_data)[c_it].is_isolated = true;
//  }

  // ---- PHASE TWO : ZOOMED SEGMENTATION ----

  pcl::io::savePCDFileBinary ("temp_oc.pcd", *cloud_octree);

  for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
    if (!(*clusters_data)[c_it].is_isolated)
    {
      float color = 512.0 + 2550.0 * rand () / (RAND_MAX + 1.0f);
      for (size_t ci_it = 0; ci_it < (*clusters_data)[c_it].indices->size (); ++ci_it)
        (*cloud_in)[(*(*clusters_data)[c_it].indices)[ci_it]].intensity = color;
    }
  pcl::io::savePCDFileBinary ("temp.pcd", *cloud_in);
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
