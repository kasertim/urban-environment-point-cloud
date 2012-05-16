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

#include <pcl/pcl_base.h>
#include <pcl/common/pca.h>
#include "extract_clusters.hpp"

// Number module
inline float
module (float a);

// Calculates the average adjacency in the octree representation
// Could be representative for the eigenvalues of PCA
// A high number / dimensionality should mainly occur for leaves, hopefully regardless of amount of leaves in cluster
float
octree_adjacency (const pcl::PointCloud<PointType>::Ptr cloud, const pcl::PointIndices indices, float resolution);

// Extract the cardinality of cluster
inline int
cardinality (const pcl::PointIndices indices);

// Extract the mean intensity value of a cluster
inline double
mean_intensity (const pcl::PointCloud<PointType>::Ptr cloud, const pcl::PointIndices indices);

// Extract clusters the EigenValue decomposition module
double
EVD (const pcl::PointCloud<PointType>::Ptr cloud,  const pcl::PointIndices indices);

// Determines the point density whitin a bounding box
double
density (const pcl::PointCloud<PointType>::Ptr cloud, const pcl::PointIndices indices);

// Extract clusters Principal component analisys
Eigen::Vector3f
pca (const pcl::PointCloud<PointType>::Ptr cloud,  const pcl::PointIndices indices);

/** \brief Estimates features for each cluster, required for classifying each cluster.
  * \param[in] cloud_in A pointer to the input point cloud.
  * \param[in/out] clusters_data An array of information holders for each cluster
  */
void
gatherClusterInformation (const pcl::PointCloud<PointType>::Ptr cloud_in, boost::shared_ptr<std::vector<ClusterData> > &clusters_data)
{
  // Creating the KdTree object for the search method of the extraction
  pcl::search::Search<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
  // For each cluster, sub_clustering is calculated
  pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
  ec.setClusterTolerance (50);
  ec.setSearchMethod (tree);
  ec.setInputCloud (global_data.cloud_octree);

  for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
  {
    std::vector<pcl::PointIndices> sub_indices;
    ec.setIndices ( (*clusters_data) [c_it].octree_indices);
    //ec.extract (sub_indices);
    ec.extract ((*clusters_data) [c_it].sub_indices);

    // convert octree indices to cloud indices
//     (*clusters_data) [c_it].sub_indices.resize (sub_indices.size ());
//     for (size_t si_it = 0; si_it < sub_indices.size (); ++si_it)
//     {
//       for (size_t si_i_it = 0; si_i_it < sub_indices[si_it].indices.size (); ++si_i_it)
//       {
//         std::vector<int> voxel_indices;
//         global_data.octree.voxelSearch (global_data.cloud_octree->points[ sub_indices[si_it].indices[si_i_it] ], voxel_indices);
//         (*clusters_data) [c_it].sub_indices[si_it].indices.insert (
//           (*clusters_data) [c_it].sub_indices[si_it].indices.end (),
//           voxel_indices.begin (), voxel_indices.end ()
//         );
//       }
//     }
    
  }

  //For each cluster, point data features are calculates
  for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
  {
    (*clusters_data) [c_it].sub_features.clear (); // Reset the current features
    (*clusters_data) [c_it].sub_features.resize ( (*clusters_data) [c_it].sub_indices.size ());
    for (size_t sc_it = 0; sc_it < (*clusters_data) [c_it].sub_indices.size (); ++sc_it)
    {
      pcl::SVMDataPoint data; // Temp object

      // Save cardinality
      data.idx = 0;
      data.value = cardinality ( (*clusters_data) [c_it].sub_indices[sc_it]);
      (*clusters_data) [c_it].sub_features[sc_it].SV.push_back (data);

      // Save mean intensity
      data.idx = 1;
      data.value = mean_intensity (global_data.cloud_octree, (*clusters_data) [c_it].sub_indices[sc_it]);
      (*clusters_data) [c_it].sub_features[sc_it].SV.push_back (data);

      // Eigen Value Decomposition module
      data.idx = 2;
      data.value = EVD (global_data.cloud_octree, (*clusters_data) [c_it].sub_indices[sc_it]);
      if (data.value != 0.0 && std::isfinite (data.value))
        (*clusters_data) [c_it].sub_features[sc_it].SV.push_back (data);

      // Extract Principal Component analisys and save the first two normalized eigenvalues
      Eigen::Vector3f eig;
      eig = pca (global_data.cloud_octree, (*clusters_data) [c_it].sub_indices[sc_it]);

      data.idx = 3;
      data.value = eig[0];
      if (data.value != 0.0 && std::isfinite (data.value))
        (*clusters_data) [c_it].sub_features[sc_it].SV.push_back (data);

      data.idx = 4;
      data.value = eig[1];
      if (data.value != 0.0 && std::isfinite (data.value))
        (*clusters_data) [c_it].sub_features[sc_it].SV.push_back (data);

      // Average octree adjacency of points
      // The resolution is the important parameter here: in the paper they vary this parameter and let the SVM select
      // the most appropriate. We could also suffice with a fixed scale I hope.
      data.idx = 5;
      data.value = octree_adjacency (global_data.cloud_octree, (*clusters_data) [c_it].sub_indices[sc_it], 0.07 * global_data.scale);
      (*clusters_data) [c_it].sub_features[sc_it].SV.push_back (data);

      // Save point density inside a bounding box
//     data.idx = 3;
//     data.value = density (cloud_in, (*clusters_data)[c_it].indices);
//     if (data.value != 0.0 && std::isfinite (data.value))
//       (*clusters_data)[c_it].features.SV.push_back (data);
    }
  }
}

float
octree_adjacency (const pcl::PointCloud<PointType>::Ptr cloud, const pcl::PointIndices indices, float resolution)
{
  // octree for downsampling
//   pcl::octree::OctreePointCloudSearch<PointType> octree (resolution);
//   pcl::IndicesPtr p_indices (new std::vector<int>);
//   p_indices->insert(p_indices->begin(), indices.indices.begin(), indices.indices.end());
//   octree.setInputCloud (cloud, p_indices );
//   octree.addPointsFromInputCloud ();

  // Results in cloud_octree
  pcl::PointCloud<PointType>::Ptr cloud_octree (new pcl::PointCloud<PointType>);
  //octree.getOccupiedVoxelCenters (cloud_octree->points);
//   cloud_octree->width = cloud_octree->points.size ();
//   cloud_octree->height = 1;
  pcl::copyPointCloud(*cloud, indices.indices, *cloud_octree);

  // kdtree for searching
  pcl::search::KdTree<PointType>::Ptr searcher (new pcl::search::KdTree<PointType>);
  searcher->setInputCloud (cloud_octree);
  std::vector<int> nn_indices;
  std::vector<float> nn_distances;

  // Search radius is crucial:
  // 1.01 * resolution -- A maximum of 6 neighbors can be found (only orthogonally adjacent)
  // 1.42 * resolution -- A maximum of 18 neighbors can be found (also planar diagonally adjacent)
  // 1.74 * resolution -- A maximum of 26 neighbors can be found (also volumetric diagonally adjacent)
  float output = 0.0;
  for (size_t p_it = 0; p_it < cloud_octree->width; ++p_it)
    output += (float) searcher->radiusSearch (p_it, 1.42 * resolution, nn_indices, nn_distances);

  // Compute the average
  return (output / (float) cloud_octree->width);
}

inline float
module (float a)
{
  if (a > 0)
    return a;
  else
    return -a;
}

inline int
cardinality (const pcl::PointIndices indices)
{
  return indices.indices.size ();
}

inline double
mean_intensity (const pcl::PointCloud<PointType>::Ptr cloud_, const pcl::PointIndices indices)
{
  int buff = 0;
  for (int j = 0; j < indices.indices.size (); j++)
  {
    buff = buff + cloud_->points[indices.indices[j]].intensity;
  }

  return (double) (buff / indices.indices.size ());
}

double
EVD (const pcl::PointCloud<PointType>::Ptr cloud_, const pcl::PointIndices indices)
{
  Eigen::Vector3f eigenVal;
  Eigen::Vector4f centroid;
  Eigen::Matrix3f covMat;

  // Compute covariance matrix
  pcl::computeMeanAndCovarianceMatrix (*cloud_, indices, covMat, centroid);

  // Compute eigenvectors
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver (covMat);
  eigenVal[0] = eigensolver.eigenvalues () [0];
  eigenVal[1] = eigensolver.eigenvalues () [1];
  eigenVal[2] = eigensolver.eigenvalues () [2];

  // Compute sqrt( l1^2 + l2^2 + l3^2 )
  return (double) sqrt (pow (eigenVal[0], 2) + pow (eigenVal[1], 2) + pow (eigenVal[2], 2));

}

double
density (const pcl::PointCloud<PointType>::Ptr cloud_, const pcl::PointIndices indices)
{
  Eigen::Vector4f minPt, maxPt;
  float volume = 0;

  // Calculate min and max point distances coordinates
  pcl::getMinMax3D (*cloud_, indices, minPt, maxPt);

  // Calculate volume
  volume = module (maxPt[0] - minPt[0]) * module (maxPt[1] - minPt[1]) * module (maxPt[2] - minPt[2]);

  // Return density
  return (double) (volume / indices.indices.size ());
}

Eigen::Vector3f
pca (const pcl::PointCloud<PointType>::Ptr cloud_,  const pcl::PointIndices indices)
{

  pcl::PCA<pcl::PointXYZI> pca;
  Eigen::Vector3f eigenVal (0.0, 0.0, 0.0);
  int sum = 0;
  
  pcl::IndicesPtr p_indices (new std::vector<int>);
  p_indices->insert(p_indices->begin(), indices.indices.begin(), indices.indices.end());

  pca.setInputCloud (cloud_);
  pca.setIndices (p_indices);

  // If the cluster is bigger than 2 points, it computer the pca.
  // Otherwise it returns a nulla vector
  if (indices.indices.size () > 2)
    eigenVal = pca.getEigenValues ();
  else
    return eigenVal;

  sum = eigenVal[0] + eigenVal[1] + eigenVal[2];

  //It returns the principal normalized eigenvalues
  return eigenVal / sum;
}
