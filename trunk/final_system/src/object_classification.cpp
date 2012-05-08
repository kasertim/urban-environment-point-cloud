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

#include "../svm_wrapper.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <fstream>

// TODO: We have only few cluster to work on. The classifier need more samples to perform a better training. We will prepare a k-nearest search to avoid the problem.

// Initialize the Visualizer
void
initVisualizer (pcl::visualization::PCLVisualizer &viewer);

// Callback of the mouse input
void
pp_callback (const pcl::visualization::PointPickingEvent &event, void *point);

// TODO The function checks for the existence of a classifier model inside the main directory.
// TODO If it fails to load the model, it start a new training procedure of the classifier.
// TODO In this version of the program, there is only one kind of noise (no distinction between vegetation and ghosts)

/** \brief The machine learning classifier, results are stored in the ClusterData structs.
  * \param[in] cloud_in A pointer to the input point cloud.
  * \param[in/out] clusters_data An array of information holders for each cluster
  */
void
applyObjectClassification (const pcl::PointCloud<PointType>::Ptr cloud_in, boost::shared_ptr<std::vector<ClusterData> > &clusters_data)
{
  // Set up the machine learnin class
  pcl::SVMTrain ml_svm_training; // To train the classifier
  pcl::SVMClassify ml_svm_classify; // To classify

  std::vector<pcl::SVMData> featuresSet; // Create the input vector for the SVM class
  std::vector<std::vector<double> > predictionOut; // Prediction output vector
  // If the input model_filename exists, it starts the classification.
  // Otherwise it starts a new machine learning training.
  if (global_data.model.size() > 0)
  {
    if (!ml_svm_classify.loadClassifierModel (global_data.model.data()))
      return;
    pcl::console::print_highlight (stderr, "Loaded ");
    pcl::console::print_value (stderr, "%s ", global_data.model.data());

    // Copy the input vector for the SVM classification
    for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
      featuresSet.push_back ( (*clusters_data) [c_it].features);

    ml_svm_classify.setInputTrainingSet (featuresSet); // Set input clusters set
    ml_svm_classify.saveNormClassProblem ("data_input"); //Save clusters features
    ml_svm_classify.setProbabilityEstimates (1); // Estimates the probabilities
    ml_svm_classify.classification ();
  }
  else
  {
    // Currently: analyze on voxels of 0.08 x 0.08 x 0.08 meter with slight alteration based on cluster aggressiveness
    float resolution = 0.08 * global_data.scale / pow (0.5 + global_data.cagg, 2);
    // Create the viewer
    pcl::visualization::PCLVisualizer viewer ("cluster viewer");

    // Output classifier model name
    global_data.model.assign (global_data.cloud_name.data());
    global_data.model.append (".model");

    std::vector<bool> lab_cluster;// save whether a cluster is labelled
    std::vector<int> pt_clst_pos; // used to memorize in the total cloud, the point affiliation to the original cluster

    // fill the vector (1 = labelled, 0 = unlabelled)
    lab_cluster.resize ( (std::size_t) clusters_data->size ());
    for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
    {
      if ( (*clusters_data) [c_it].is_isolated)
      {
        (*clusters_data) [c_it].features.label = 0;
        lab_cluster[c_it] = 1;
      }
      else
        lab_cluster[c_it] = 0;
    }

    // Build a cloud with unlabelled clusters
    pcl::PointCloud<PointType>::Ptr fragm_cloud (new pcl::PointCloud<PointType>);

    // Initialize the viewer
    initVisualizer (viewer);
    PointType picked_point; // changed whether a mouse click occours. It saves the selected cluster index
    viewer.registerPointPickingCallback (&pp_callback, (void *) &picked_point);
    
    // Create a cloud with unlabelled clusters
    for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
      if (!lab_cluster[c_it])
      {
        pcl::PointCloud<PointType>::Ptr cluster (new pcl::PointCloud<PointType>);
        pcl::copyPointCloud (*cloud_in, * (*clusters_data) [c_it].indices, *cluster);
	
	// Downsample cluster
	pcl::VoxelGrid<PointType> sor;
	sor.setInputCloud (cluster);
        sor.setLeafSize (resolution, resolution, resolution);
	sor.filter (*cluster);
	
	// Copy cluster into a global cloud
        fragm_cloud->operator+= (*cluster);

        // Fill a vector to memorize the original affiliation of a point to the cluster
        for (int clust_pt = 0; clust_pt < cluster->size(); clust_pt++)
          pt_clst_pos.push_back (c_it);

        // Add cluster to the viewer
        std::stringstream cluster_name;
        cluster_name << "cluster" << c_it;
        pcl::visualization::PointCloudColorHandlerGenericField<PointType> rgb (cluster, "intensity");// Get color handler for the cluster cloud
        viewer.addPointCloud<PointType> (cluster, rgb, cluster_name.str().data());
      }
    
    // Create a tree for point searching in the total cloud
    pcl::KdTreeFLANN<pcl::PointXYZI> tree_;
    tree_.setInputCloud (fragm_cloud);

    // Visualize the whole cloud
    while (!viewer.wasStopped())
    {
      if (picked_point.x != 0.0f || picked_point.y != 0.0f || picked_point.z != 0.0f)  // if a point is clicked
      {
//          cout << picked_point.x << " " << picked_point.y << " " << picked_point.z << endl;
        std::vector<int> pointIdxNKNSearch (1);
        std::vector<float> pointNKNSquaredDistance (1);
        tree_.nearestKSearch (picked_point, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
//         cout << fragm_cloud->points[pointIdxNKNSearch[0]].x
//              << " " << fragm_cloud->points[pointIdxNKNSearch[0]].y
//              << " " << fragm_cloud->points[pointIdxNKNSearch[0]].z
//              << endl;
//         cout << pt_clst_pos[pointIdxNKNSearch[0]] << endl << endl;

        std::stringstream cluster_name;
        cluster_name << "cluster" << pt_clst_pos[pointIdxNKNSearch[0]];
        lab_cluster[ pt_clst_pos[pointIdxNKNSearch[0]] ] = 1; // cluster is marked as labelled
        (*clusters_data) [pt_clst_pos[pointIdxNKNSearch[0]]].features.label = 1; // the cluster is set as a noise
        viewer.removePointCloud (cluster_name.str().data());

        picked_point.x = 0.0f;
        picked_point.y = 0.0f;
        picked_point.z = 0.0f;
      }
      boost::this_thread::sleep (boost::posix_time::microseconds (100000));
      viewer.spinOnce (500);
    }

    // Close the viewer
    viewer.close();

    // The remaining unlabelled clusters are marked as "good"
    for (int c_it = 0; c_it < lab_cluster.size(); c_it++)
    {
      if (!lab_cluster[c_it])
        (*clusters_data) [c_it].features.label = 0; // Mark remaining clusters as good
    }

    // Copy the input vector for the SVM classification
    for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
      featuresSet.push_back ( (*clusters_data) [c_it].features);

    // Setting the training classifier
    pcl::SVMParam trainParam;
    trainParam.probability = 1; // Estimates the probabilities
    trainParam.C = 8; // Initial C value of the classifier
    trainParam.gamma = 0.5; // Initial gamma value of the classifier

    ml_svm_training.setInputTrainingSet (featuresSet);  // Set input training set
    ml_svm_training.setParameters (trainParam);
    ml_svm_training.trainClassifier(); // Train the classifier
    ml_svm_training.saveClassifierModel (global_data.model.data()); // Save classifier model
    pcl::console::print_highlight (stderr, "Saved ");
    pcl::console::print_value (stderr, "%s ", global_data.model.data());
    ml_svm_training.saveTrainingSet ("data_input"); // Save clusters features normalized

    // Test the current classification
    ml_svm_classify.loadClassifierModel (global_data.model.data());
    ml_svm_classify.setInputTrainingSet (featuresSet);
    ml_svm_classify.setProbabilityEstimates (1);
    ml_svm_classify.classificationTest ();
  }

  ml_svm_classify.saveClassificationResult ("prediction"); // save prediction in outputtext file
  ml_svm_classify.getClassificationResult (predictionOut);

  // Get labels order
  std::vector<int> labels;
  ml_svm_classify.getLabel (labels);

  // Store the boolean output inside clusters_data
  for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
    switch ( (int) predictionOut[c_it][0])
    {
      case 0:
        (*clusters_data) [c_it].is_good = true;
        break;
      case 1:
        (*clusters_data) [c_it].is_ghost = true;
        break;
      case 2:
        (*clusters_data) [c_it].is_tree = true;
        break;
    }

  // Store the percentage output inside cluster_data
  for (size_t lab = 0; lab < labels.size (); lab++)
    switch (labels[lab])
    {
      case 0:
        for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
        {
          (*clusters_data) [c_it].is_good_prob = predictionOut[c_it][lab + 1];
        }
        break;
      case 1:
        for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
        {
          (*clusters_data) [c_it].is_ghost_prob = predictionOut[c_it][lab + 1];
        }
        break;
      case 2:
        for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
        {
          (*clusters_data) [c_it].is_tree_prob = predictionOut[c_it][lab + 1];
        }
        break;
    }
};

void
initVisualizer (pcl::visualization::PCLVisualizer &viewer)
{
  // Setting the initial viewer parameters
  viewer.initCameraParameters ();
  viewer.setBackgroundColor (0, 0, 0);
  viewer.addCoordinateSystem (1000);
  viewer.camera_.view[0] = 0;
  viewer.camera_.view[1] = 0;
  viewer.camera_.view[2] = 1;
  viewer.camera_.pos[0] = 8000;
  viewer.camera_.pos[1] = 20000;
  viewer.camera_.pos[2] = 2500;
  viewer.updateCamera ();
  viewer.addText ("Shift + click to select noisy objects", 50, 300, "user");
}

void
pp_callback (const pcl::visualization::PointPickingEvent &event, void *point)
{
  if (event.getPointIndex () == -1)
    return;
  PointType *idx;
  idx = static_cast<PointType *> (point);
  // A single point has been selected
  event.getPoint ( (*idx).x, (*idx).y, (*idx).z);
}
