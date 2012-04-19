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
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

// TODO: The min_segment_size_coefficient could be improved to no longer depend on a global input
// TODO: Maybe use a passthrough filter or some other system to check only the bottom half of the scene for ground planes

/** \brief Segments ground planes from the input cloud.
  * \param[in] cloud_in A pointer to the input point cloud.
  * \param[in/out] indices A pointer to the input/remaining indices.
  */
void
applyPlaneSegmentation (const pcl::PointCloud<PointType>::Ptr cloud_in, pcl::IndicesPtr &indices)
{
  // ---- AUTOMATED INPUT PARAMETERS OF THIS SECTION ----
  // Points within this distance from a calculated plane will be marked as belonging to that plane
  // Setting this larger will be more useful for segmenting surfaces that are not perfectly flat
  // It will however also start removed larger portions of objects that are adjacent to that surface
  // Currently: assume all points within 0.06 meters distance on either side of the plane
  float distance_threshold = 0.06 * global_data.scale;
  // Speed / accuracy tradeoff for this section, also used for equalizing density
  // Note that the gain in accuracy can be neglected when this is significantly smaller than distance_threshold
  // Currently: analyze on 40% of distance_threshold
  float resolution = 0.4 * distance_threshold;
  // The number of points in the octree representation gets multiplied by this coefficient to get to minimum plane segment size
  float min_segment_size_coefficient = global_data.pground;

  // An octree representation class for temporary downsampling
  pcl::octree::OctreePointCloudSearch<PointType> octree (resolution);
  octree.setInputCloud (cloud_in, indices);
  octree.addPointsFromInputCloud ();

  // A RANSAC planar segmentation class
  pcl::SACSegmentation<PointType> sacs;
  sacs.setMethodType (pcl::SAC_RANSAC);
  sacs.setModelType (pcl::SACMODEL_PERPENDICULAR_PLANE);
//  sacs.setModelType (pcl::SACMODEL_PLANE);
//  sacs.setOptimizeCoefficients (true);
  sacs.setDistanceThreshold (distance_threshold);
  sacs.setMaxIterations (300);
  sacs.setAxis (Eigen::Vector3f (0.0, 0.0, 1.0));
  sacs.setEpsAngle (0.314);

  // An ExtractIndices class for inverting indices
  pcl::ExtractIndices<PointType> ei;
  ei.setNegative (true);

  // Variables used
  pcl::PointCloud<PointType>::Ptr cloud_octree (new pcl::PointCloud<PointType>);
  pcl::ModelCoefficients coefficients;
  pcl::PointIndices current_plane;
  pcl::IndicesPtr removed_points (new std::vector<int>);
  pcl::IndicesPtr remaining_points (new std::vector<int>);
  float color = 0.0;

  // Downsample and pass the bottom
  octree.getOccupiedVoxelCenters (cloud_octree->points);
  cloud_octree->width = cloud_octree->points.size ();
  cloud_octree->height = 1;
  sacs.setInputCloud (cloud_octree);
  ei.setInputCloud (cloud_octree);

  // First segmentation to enter loop properly, the first plane is also not restricted by min_segment_size
  sacs.segment (current_plane, coefficients);

  // Loop while the found plane is large enough
  do
  {
    // Color the plane for intermediate output
    if (global_data.octrees)
    {
      color += 1.0;
      for (size_t i_it = 0; i_it < current_plane.indices.size (); ++i_it)
        cloud_octree->points[current_plane.indices[i_it]].intensity = color;
    }

    // Concatenate all found planes into removed_points
    (*removed_points).insert ((*removed_points).end (), current_plane.indices.begin (), current_plane.indices.end ());

    // Extract all indices *except* the ones of the planes found so far
    remaining_points->clear ();
    ei.setIndices (removed_points);
    ei.filter (*remaining_points);

    // Re-run segmentation on the remaining points
    sacs.setIndices (remaining_points);
    sacs.segment (current_plane, coefficients);
  }
  while (current_plane.indices.size () > min_segment_size_coefficient * cloud_octree->width);

  if (removed_points->size () > 0)
  {
    // Upsample back from octree representation and store in output
    indices->clear ();
    for (size_t i_it = 0; i_it < remaining_points->size (); ++i_it)
    {
      std::vector<int> voxel_indices;
      octree.voxelSearch (cloud_octree->points[(*remaining_points)[i_it]], voxel_indices);
      indices->insert (indices->end (), voxel_indices.begin (), voxel_indices.end ());
    }

    // Output the colored planes octree
    if (global_data.octrees)
    {
      ei.setIndices (remaining_points);
      ei.filter (*cloud_octree);
      pcl::io::savePCDFileBinary ("octree_planes.pcd", *cloud_octree);
      pcl::console::print_info (stderr, "- Saved ");
      pcl::console::print_value (stderr, "octree_planes.pcd ");
    }
  }
}
