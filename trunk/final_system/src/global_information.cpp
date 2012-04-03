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

// Information holder for the full point cloud
struct GlobalData
{
  pcl::IndicesPtr indices;
  float x_min, x_max, y_min, y_max, z_min, z_max, i_min, i_max;
  float cagg;
  GlobalData () :
      indices (new std::vector<int>)
  {
    x_min = y_min = z_min = i_min = std::numeric_limits<float>::max ();
    x_max = y_max = z_max = i_max = std::numeric_limits<float>::min ();
  }
};

/** \brief Gathers information about the entire input cloud, useful for the other pipeline stages: automation/normalization of input parameters.
 * \param[in] cloud_in A pointer to the input point cloud.
 * \param[out] global_data A struct holding information on the full point cloud and global input parameters.
 */
void
gatherGlobalInformation (const pcl::PointCloud<PointType>::Ptr cloud_in, GlobalData global_data)
{
  for (int p_it = 0; p_it < static_cast<int> (cloud_in->points.size ()); ++p_it)
  {
    if (cloud_in->points[p_it].x < global_data.x_min)
      global_data.x_min = cloud_in->points[p_it].x;
    if (cloud_in->points[p_it].x > global_data.x_max)
      global_data.x_max = cloud_in->points[p_it].x;
    if (cloud_in->points[p_it].y < global_data.y_min)
      global_data.y_min = cloud_in->points[p_it].y;
    if (cloud_in->points[p_it].y > global_data.y_max)
      global_data.y_max = cloud_in->points[p_it].y;
    if (cloud_in->points[p_it].z < global_data.z_min)
      global_data.z_min = cloud_in->points[p_it].z;
    if (cloud_in->points[p_it].z > global_data.z_max)
      global_data.z_max = cloud_in->points[p_it].z;
    if (cloud_in->points[p_it].intensity < global_data.i_min)
      global_data.i_min = cloud_in->points[p_it].intensity;
    if (cloud_in->points[p_it].intensity > global_data.i_max)
      global_data.i_max = cloud_in->points[p_it].intensity;
  }
}
