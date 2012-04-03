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

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

// TODO: This step gets increasingly slower for larger data sets, maybe use an octree/voxelgrid to downsample temporarily
// TODO: Check the model coefficients that are output after each segmentation and make sure you get rid of grounds (or walls) only
// TODO: The min_segment_size parameter needs to be altered more adaptively
// TODO: Maybe use different min_segment_size parameters: for walls use higher values than for ground planes

/** \brief Segments ground and wall planes from the input cloud.
  * \param[in] cloud_in A pointer to the input point cloud.
  * \param[in/out] global_data A struct holding information on the full point cloud and global input parameters.
  */
void
applyPlaneSegmentation (const pcl::PointCloud<PointType>::Ptr cloud_in,
                        GlobalData &global_data)
{
  // Input parameters for this section:
  int min_segment_size = global_data.cardinality / 10; // Maybe make this dependent off a global input parameter for cluttering
  float distance_threshold = 0.1 * global_data.scale; // 10 centimeters of noise allowed

  // A RANSAC segmentation class
  pcl::SACSegmentation<PointType> sacs;
  sacs.setInputCloud (cloud_in);
  sacs.setMethodType (pcl::SAC_RANSAC);
  sacs.setModelType (pcl::SACMODEL_PLANE);
  sacs.setOptimizeCoefficients (true);
  sacs.setDistanceThreshold (distance_threshold);
  sacs.setMaxIterations (100);

  // An ExtractIndices class for inverting indices
  pcl::ExtractIndices<PointType> ei;
  ei.setInputCloud (cloud_in);
  ei.setNegative (true);

  // Variables used
  pcl::ModelCoefficients coefficients;
  pcl::PointIndices current_plane;
  pcl::IndicesPtr removed_points (new std::vector<int>);
  pcl::IndicesPtr remaining_points (new std::vector<int>);

  // First segmentation to enter loop properly
  sacs.segment (current_plane, coefficients);

  // Loop while the found plane is large enough
  while (current_plane.indices.size () > min_segment_size)
  {
    // Concatenate all planes into removed_points
    (*removed_points).insert ((*removed_points).end (), current_plane.indices.begin (), current_plane.indices.end ());

    // Extract all indices *except* the ones of the planes
    remaining_points->clear ();
    ei.setIndices (removed_points);
    ei.filter (*remaining_points);

    // Re-run segmentation on the remaining points
    sacs.setIndices (remaining_points);
    sacs.segment (current_plane, coefficients);
  }

  // Extract all indices *except* the ones of the planes for output
  ei.setIndices (removed_points);
  ei.filter (*global_data.indices);
}
