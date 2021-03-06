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

#include <pcl/filters/extract_indices.h>

/** \brief This will be a simple ExtractIndices application for now, but for more elaborate noise it could mean smoothing or stuff like that.
  * \param[in] cloud_in A pointer to the input point cloud.
  * \param[in] clusters_data An array of information holders for each cluster
  * \param[out] cloud_out The output point cloud, with the noise filtered.
  */
void
applyNoiseFiltering (const pcl::PointCloud<PointType>::Ptr cloud_in, boost::shared_ptr<std::vector<ClusterData> > clusters_data,
                     pcl::PointCloud<PointType>::Ptr &cloud_out)
{
  // Set up the noise index array
  pcl::IndicesPtr noise_indices (new std::vector<int>);

  // Fill in the index array with all noise classified objects
  for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
    if ((*clusters_data)[c_it].is_isolated || (*clusters_data)[c_it].is_tree || (*clusters_data)[c_it].is_ghost)
      noise_indices->insert (noise_indices->end (), (*clusters_data)[c_it].indices->begin (), (*clusters_data)[c_it].indices->end ());

  // Remove the corresponding points from the cloud
  pcl::ExtractIndices<PointType> ei;
  ei.setInputCloud (cloud_in);
  ei.setIndices (noise_indices);
  ei.setNegative (true);
  if (cloud_in->isOrganized ())
    ei.setKeepOrganized (true);
  ei.filter (*cloud_out);
}
