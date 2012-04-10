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

// Number module
inline float module(float a);

// Extract the cardinality of cluster
inline int cardinality(pcl::IndicesPtr indices_);

// Extract the mean intensity value of a cluster
inline double mean_intensity(const pcl::PointCloud<PointType>::Ptr cloud_ ,pcl::IndicesPtr indices_);

// Extract clusters the EigenValue decomposition module
double EVD(const pcl::PointCloud<PointType>::Ptr cloud_ ,pcl::IndicesPtr indices_);

// Determines the point density whitin a bounding box
double density(const pcl::PointCloud<PointType>::Ptr cloud_ ,pcl::IndicesPtr indices_);


// Extract clusters Principal component analisys
Eigen::Vector3f pca(const pcl::PointCloud<PointType>::Ptr cloud_ ,pcl::IndicesPtr indices_);

/** \brief Estimates features for each cluster, required for classifying each cluster.
  * \param[in] cloud_in A pointer to the input point cloud.
  * \param[in] clusters_data An array of information holders for each cluster
  * \param[out] global_data A struct holding information on the full point cloud, global input parameters and clusters features.
  */
void
gatherClusterInformation (const pcl::PointCloud<PointType>::Ptr cloud_in,
                          boost::shared_ptr<std::vector<ClusterData> > &clusters_data,
                          GlobalData &global_data)
{
  // Calculated features will be saves inside global_data.features
    global_data.features.clear();
    global_data.features.resize (clusters_data->size ());

    //For each cluster, point data features are calculates
    for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
    {
        pcl::svmDataPoint data; // Temp object
	
	// Save carcinality
	data.idx = 0;
	data.value = cardinality((*clusters_data)[c_it].indices);
	global_data.features[c_it].SV.push_back(data);
	
	// Save mean intensity
	data.idx = 1;
	data.value = mean_intensity(cloud_in, (*clusters_data)[c_it].indices);
	global_data.features[c_it].SV.push_back(data);
	
	// Eigen Value Decomposition module
	data.idx = 2;
	data.value = EVD(cloud_in, (*clusters_data)[c_it].indices);
	if(data.value != 0.0 && std::isfinite(data.value) ) 
	  global_data.features[c_it].SV.push_back(data);
	
	// Save point density inside a bounding box
	data.idx = 3;
	data.value = density(cloud_in, (*clusters_data)[c_it].indices);
	if(data.value != 0.0 && std::isfinite(data.value) ) 
	  global_data.features[c_it].SV.push_back(data);
	
	// Extract Principal Component analisys and save the first two normalized eigenvalues
	Eigen::Vector3f eig;
	eig = pca(cloud_in, (*clusters_data)[c_it].indices);
	
	data.idx = 4;
	data.value = eig[0];
	if(data.value != 0.0 && std::isfinite(data.value) ) 
	  global_data.features[c_it].SV.push_back(data);
	
	data.idx = 5;
	data.value = eig[1];
	if(data.value != 0.0 && std::isfinite(data.value) ) 
	  global_data.features[c_it].SV.push_back(data);

    }
}

inline float module(float a) {
    if (a>0)
        return a;
    else
        return -a;
}

inline int cardinality(pcl::IndicesPtr indices_) {
    return indices_->size();
}

inline double mean_intensity(const pcl::PointCloud<PointType>::Ptr cloud_ ,pcl::IndicesPtr indices_) {
    int buff=0;
    for (int j=0; j < indices_->size(); j++)
    {
        buff = buff + cloud_->points[ indices_->operator[](j) ].intensity;
    }

    return  (double)(buff / indices_->size());
}

double EVD(const pcl::PointCloud<PointType>::Ptr cloud_ ,pcl::IndicesPtr indices_) {
    Eigen::Vector3f eigenVal;
    Eigen::Vector4f centroid;
    Eigen::Matrix3f covMat;

    // Compute covariance matrix
    pcl::computeMeanAndCovarianceMatrix (*cloud_ , *indices_, covMat, centroid);

    // Compute eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(covMat);
    eigenVal[0] = eigensolver.eigenvalues()[0];
    eigenVal[1] = eigensolver.eigenvalues()[1];
    eigenVal[2] = eigensolver.eigenvalues()[2];

    // Compute sqrt( l1^2 + l2^2 + l3^2 )
    return (double)sqrt( pow(eigenVal[0],2) + pow(eigenVal[1],2) + pow(eigenVal[2],2) );

}

double density(const pcl::PointCloud<PointType>::Ptr cloud_ ,pcl::IndicesPtr indices_) {
    Eigen::Vector4f minPt, maxPt;
    float volume =0;
    
    // Calculate min and max point distances coordinates
    pcl::getMinMax3D(*cloud_, *indices_, minPt, maxPt);

    
    // Calculate volume
    volume = module( maxPt[0] - minPt[0] ) * module( maxPt[1] - minPt[1] ) * module( maxPt[2] - minPt[2]);

    // Return density
    return (double)(volume / indices_->size());
}

Eigen::Vector3f pca(const pcl::PointCloud<PointType>::Ptr cloud_ ,pcl::IndicesPtr indices_) {

    pcl::PCA<pcl::PointXYZI> pca;
    Eigen::Vector3f eigenVal(0.0, 0.0, 0.0);
    int sum=0;

    pca.setInputCloud(cloud_);
    pca.setIndices(indices_);
    
    // If the cluster is bigger than 2 points, it computer the pca. 
    // Otherwise it returns a nulla vector
    if (indices_->size() > 2)
        eigenVal = pca.getEigenValues();
    else
        return eigenVal;

    sum = eigenVal[0]+eigenVal[1]+eigenVal[2];

    //It returns the principal normalized eigenvalues
    return eigenVal / sum;
}