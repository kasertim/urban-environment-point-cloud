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

// TODO: The min_segment_size_coefficient could be improved to adapt to the amount of expected contents in the scene
// TODO: Maybe only apply to grounds instead of any large plane

/** \brief Segments ground and wall planes from the input cloud.
  * \param[in] cloud_in A pointer to the input point cloud.
  * \param[in/out] global_data A struct holding information on the full point cloud and global input parameters.
  */
void
applyPlaneSegmentation (const pcl::PointCloud<PointType>::Ptr cloud_in,
                        GlobalData &global_data)
{
  // ---- AUTOMATED INPUT PARAMETERS OF THIS SECTION ----
  // Points within this distance from a calculated plane will be marked as belonging to that plane
  // Setting this larger will be more useful for segmenting surfaces that are not perfectly flat
  // It will however also start removed larger portions of objects that are adjacent to that surface
  // Currently: assume all points within 0.2 meters distance on either side of the plane
  float distance_threshold = 0.2 * global_data.scale;
  // Speed / accuracy tradeoff for this section, also used for equalizing density
  // Note that the gain in accuracy can be neglected when this is significantly smaller than distance_threshold
  // Currently: analyze on voxels of 0.1 x 0.1 x 0.1 meter
  float resolution = 0.5 * distance_threshold;
  // The number of points in the octree representation gets multiplied by this coefficient to get to minimum segment size
  // Maybe make this dependent off a global input parameter for the amount of expected contents in the scene
  // Currently: segment planes that make up at least 10% of the scene (after density equalization)
  float min_segment_size_coefficient = 0.1;

  // An octree representation class for temporary downsampling
  pcl::octree::OctreePointCloudSearch<PointType> octree (resolution);
  octree.setInputCloud (cloud_in, global_data.indices);
  octree.addPointsFromInputCloud ();

  // A RANSAC planar segmentation class
  pcl::SACSegmentation<PointType> sacs;
  sacs.setMethodType (pcl::SAC_RANSAC);
  sacs.setModelType (pcl::SACMODEL_PLANE);
  sacs.setOptimizeCoefficients (true);
  sacs.setDistanceThreshold (distance_threshold);
  sacs.setMaxIterations (100);

  // An ExtractIndices class for inverting indices
  pcl::ExtractIndices<PointType> ei;
  ei.setNegative (true);

  // Variables used
  pcl::PointCloud<PointType>::Ptr cloud_octree (new pcl::PointCloud<PointType>);
  pcl::ModelCoefficients coefficients;
  pcl::PointIndices current_plane;
  pcl::IndicesPtr removed_points (new std::vector<int>);
  pcl::IndicesPtr remaining_points (new std::vector<int>);

  // Downsampling
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

  // Upsample back from octree representation and store in output
  global_data.indices->clear ();
  for (size_t i_it = 0; i_it < remaining_points->size (); ++i_it)
  {
    std::vector<int> voxel_indices;
    octree.voxelSearch (cloud_octree->points[(*remaining_points)[i_it]], voxel_indices);
    global_data.indices->insert (global_data.indices->end (), voxel_indices.begin (), voxel_indices.end ());
  }
}
