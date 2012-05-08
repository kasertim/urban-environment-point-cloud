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

// TODO: The segment_size_coefficients could be improved to no longer depend on global inputs

/** \brief Divides the remaining points into several clusters, each cluster likely to contain exactly one object.
  * \param[in] cloud_in A pointer to the input point cloud.
  * \param[in] indices A pointer to the remaining indices.
  * \param[out] clusters_data An array of information holders for each cluster
  */
void
applyObjectClustering (const pcl::PointCloud<PointType>::Ptr cloud_in, const pcl::IndicesPtr indices, boost::shared_ptr<std::vector<ClusterData> > &clusters_data)
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
  // The number of points in the octree representation gets multiplied by this coefficient to get to maximum cluster size
  // Larger than this size means the cluster is part of the background and is not passed to the classifier since it is non-noise
  float max_cluster_size_coefficient = global_data.plarge;
  // The number of points in the octree representation gets multiplied by this coefficient to get to minimum cluster size
  // Smaller than this size means the cluster is isolated and is not passed to the classifier since it is noise
  float min_cluster_size_coefficient = global_data.psmall;

  // An octree representation class for temporary downsampling and density normalization
  pcl::octree::OctreePointCloudSearch<PointType> octree (resolution);
  //pcl::PointCloud<PointType>::Ptr global_data.cloud_octree (new pcl::PointCloud<PointType>);
  octree.setInputCloud (cloud_in, indices);
  octree.addPointsFromInputCloud ();
  octree.getOccupiedVoxelCenters (global_data.cloud_octree->points);
  global_data.cloud_octree->width = global_data.cloud_octree->points.size ();
  global_data.cloud_octree->height = 1;

  // A kdtree class for nearest neighbor searching
  pcl::search::KdTree<PointType>::Ptr searcher (new pcl::search::KdTree<PointType>);
  searcher->setInputCloud (global_data.cloud_octree);

  // Feature estimation: Curvature
  pcl::NormalEstimation<PointType, pcl::Normal> ne;
  pcl::PointCloud<pcl::Normal> cloud_normals;
  ne.setInputCloud (global_data.cloud_octree);
  ne.setSearchMethod (searcher);
  ne.setRadiusSearch (far_distance_threshold);
  ne.compute (cloud_normals);

  // Feature estimation: Intensity
  std::vector<int> intensity (global_data.cloud_octree->width, 0.0);
  for (size_t p_it = 0; p_it < intensity.size (); ++p_it)
  {
    std::vector<int> voxel_indices;
    octree.voxelSearch (global_data.cloud_octree->points[p_it], voxel_indices);
    for (size_t vi_it = 0; vi_it < voxel_indices.size (); ++vi_it)
      intensity[p_it] += (*cloud_in).points[voxel_indices[vi_it]].intensity;
    intensity[p_it] /= voxel_indices.size ();
  }

  // ---- CONDITIONAL REGION GROWING WITH BASIC OVER-SEGMENTATION PREVENTION ----
  // Three region growing passes are performed:
  // 1) Tags all the isolated clusters: clusters smaller than min_cluster_size
  // 2) The isolated points possibly generated due to over-segmentation are now added to a nearby big cluster
  // 3) The remaining isolated points need to get mapped into clusters for output as well

  // Region growing tags
  std::vector<bool> processed (global_data.cloud_octree->width, false);
  std::vector<bool> isolated (global_data.cloud_octree->width, false);

  // First region growing pass: Tag isolated points
  for (size_t p_it = 0; p_it < global_data.cloud_octree->width; ++p_it)
  {
    // Only iterate through potential seed points
    if (processed[p_it])
      continue;
    processed[p_it] = true;

    // Set up the seed queue
    size_t sq_it = 0;
    std::vector<int> seed_queue;
    seed_queue.push_back (p_it);
    while (sq_it < seed_queue.size ())
    {
      // Candidates in vicinity are stored in nn_indices
      std::vector<int> nn_indices;
      std::vector<float> nn_distances;
      if (searcher->radiusSearch (seed_queue[sq_it], far_distance_threshold, nn_indices, nn_distances) < 1)
      {
        sq_it++;
        continue;
      }
      for (size_t nn_it = 1; nn_it < nn_indices.size (); ++nn_it) // nn_indices[0] should be sq_it
      {
        // This candidate is already part of a cluster
        if (processed[nn_indices[nn_it]])
          continue;
        // Does this candidate satisfy the conditions to make it part of this cluster?
        // Either it is very close AND satisfies close_curvature_threshold OR close_intensity_threshold
        // Either it is not very close AND satisfies far_curvature_threshold AND far_intensity_threshold
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

    // If the found cluster is very small, tag the points as isolated
    if (seed_queue.size () < min_cluster_size_coefficient * global_data.cloud_octree->width)
      for (size_t sq_it = 0; sq_it < seed_queue.size (); ++sq_it)
        isolated[seed_queue[sq_it]] = true;
  }

  // Second region growing pass: The isolated points possibly isolated due to over-segmentation are now added to a nearby big cluster
  for (size_t p_it = 0; p_it < global_data.cloud_octree->width; ++p_it)
  {
    // Only iterate through potential seed points, isolated points are not allowed to be a seed
    if (!processed[p_it] || isolated[p_it])
      continue;
    processed[p_it] = false;

    // Set up the seed queue
    size_t sq_it = 0;
    std::vector<int> seed_queue;
    seed_queue.push_back (p_it);
    while (sq_it < seed_queue.size ())
    {
      // Candidates in vicinity are stored in nn_indices
      std::vector<int> nn_indices;
      std::vector<float> nn_distances;
      if (searcher->radiusSearch (seed_queue[sq_it], far_distance_threshold, nn_indices, nn_distances) < 1)
      {
        sq_it++;
        continue;
      }
      for (size_t nn_it = 1; nn_it < nn_indices.size (); ++nn_it) // nn_indices[0] should be sq_it
      {
        // This candidate is already part of a cluster
        if (!processed[nn_indices[nn_it]])
          continue;
        // Does this candidate satisfy the conditions to make it part of this cluster?
        // Either it is isolated
        // Either it is very close AND satisfies close_curvature_threshold OR close_intensity_threshold
        // Either it is not very close AND satisfies far_curvature_threshold AND far_intensity_threshold
        if (isolated[nn_indices[nn_it]]
            || (nn_distances[nn_it] < pow (close_distance_threshold, 2)
                && (fabs (cloud_normals.points[nn_indices[0]].curvature - cloud_normals.points[nn_indices[nn_it]].curvature)
                    < close_curvature_threshold || fabs (intensity[nn_indices[0]] - intensity[nn_indices[nn_it]]) < close_intensity_threshold))
            || (fabs (cloud_normals.points[nn_indices[0]].curvature - cloud_normals.points[nn_indices[nn_it]].curvature) < far_curvature_threshold
                && fabs (intensity[nn_indices[0]] - intensity[nn_indices[nn_it]]) < far_intensity_threshold))
        {
          seed_queue.push_back (nn_indices[nn_it]);
          processed[nn_indices[nn_it]] = false;
        }
      }
      sq_it++;
    }

    // Upsample back from octree representation and store in output
    if (seed_queue.size () < max_cluster_size_coefficient * global_data.cloud_octree->width)
    {
      ClusterData cluster;
      //float color = 2.0 + (rand () % 9);
      int color = rand () % 256;
      for (size_t sq_it = 0; sq_it < seed_queue.size (); ++sq_it)
      {
        global_data.cloud_octree->points[seed_queue[sq_it]].intensity = color;
        std::vector<int> voxel_indices;
        octree.voxelSearch (global_data.cloud_octree->points[seed_queue[sq_it]], voxel_indices);
        cluster.indices->insert (cluster.indices->end (), voxel_indices.begin (), voxel_indices.end ());
      }
      clusters_data->push_back (cluster);
    }
  }

  // Third region growing pass: The remaining isolated points need to get mapped into clusters as well
  for (size_t p_it = 0; p_it < global_data.cloud_octree->width; ++p_it)
  {
    // Only iterate through the remaining unprocessed points, which are all truly isolated ones
    if (!processed[p_it])
      continue;
    processed[p_it] = false;

    // Set up the seed queue
    size_t sq_it = 0;
    std::vector<int> seed_queue;
    seed_queue.push_back (p_it);
    while (sq_it < seed_queue.size ())
    {
      // Candidates in vicinity are stored in nn_indices
      std::vector<int> nn_indices;
      std::vector<float> nn_distances;
      if (searcher->radiusSearch (seed_queue[sq_it], far_distance_threshold, nn_indices, nn_distances) < 1)
      {
        sq_it++;
        continue;
      }
      for (size_t nn_it = 1; nn_it < nn_indices.size (); ++nn_it) // nn_indices[0] should be sq_it
      {
        // This candidate is already part of a cluster
        if (!processed[nn_indices[nn_it]])
          continue;
        // Does this candidate satisfy the conditions to make it part of this cluster?
        // Either it is very close AND satisfies close_curvature_threshold OR close_intensity_threshold
        // Either it is not very close AND satisfies far_curvature_threshold AND far_intensity_threshold
        if ((nn_distances[nn_it] < pow (close_distance_threshold, 2)
            && (fabs (cloud_normals.points[nn_indices[0]].curvature - cloud_normals.points[nn_indices[nn_it]].curvature) < close_curvature_threshold
                || fabs (intensity[nn_indices[0]] - intensity[nn_indices[nn_it]]) < close_intensity_threshold))
            || (fabs (cloud_normals.points[nn_indices[0]].curvature - cloud_normals.points[nn_indices[nn_it]].curvature) < far_curvature_threshold
                && fabs (intensity[nn_indices[0]] - intensity[nn_indices[nn_it]]) < far_intensity_threshold))
        {
          seed_queue.push_back (nn_indices[nn_it]);
          processed[nn_indices[nn_it]] = false;
        }
      }
      sq_it++;
    }

    // Upsample back from octree representation and store in output
    ClusterData cluster;
    for (size_t sq_it = 0; sq_it < seed_queue.size (); ++sq_it)
    {
      std::vector<int> voxel_indices;
      octree.voxelSearch (global_data.cloud_octree->points[seed_queue[sq_it]], voxel_indices);
      cluster.indices->insert (cluster.indices->end (), voxel_indices.begin (), voxel_indices.end ());
    }
    cluster.is_isolated = true;
    clusters_data->push_back (cluster);
  }

  // Output the colored clusters octree
  if (global_data.octrees)
  {
    pcl::io::savePCDFileBinary ("octree_clusters.pcd", *global_data.cloud_octree);
    pcl::console::print_info (stderr, "- Saved ");
    pcl::console::print_value (stderr, "octree_clusters.pcd ");
     system("pcl_pcd_viewer -multiview 1 octree_clusters.pcd octree_planes.pcd") ;
  }
}
