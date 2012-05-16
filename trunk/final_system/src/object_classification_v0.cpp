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
#include <boost/thread/thread.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <fstream>

// TODO: We have only few cluster to work on. The classifier need more samples to perform a better training. We will prepare a k-nearest search to avoid the problem.

// Display single cluster asking for an user input. It can be 0 for good cluster, 1 for ghosts, 2 for trees.
int
getInputLabel (const pcl::PointCloud<PointType>::Ptr cloud_in, pcl::IndicesPtr indices_, int i, int n_clusters_,
               boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer);

// Initialize the Visualizer
void
initVisualizer (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, 
		const pcl::PointCloud<PointType>::Ptr cloud_in);

// Get an input key pressed, and store in stop_void whether 0, 1 or 2 is pressed
void
keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* stop_void);

// Free the buffer
void
deleteBuffer (const pcl::visualization::KeyboardEvent &event, void* stop_void);

// TODO The function checks for the existence of a classifier model inside the main directory. If it fails to load the model, it start a new training procedure of the classifier.

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

//     FILE.close();

  }
  else
  {
    // Output classifier model name
    global_data.model.assign (global_data.cloud_name.data() );
    global_data.model.append (".model");
    
    // Initialize the viewer
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer 
      (new pcl::visualization::PCLVisualizer ("Cluster Viewer"));
    
    initVisualizer (viewer, cloud_in);

    // Checks an user input for each cluster. Stores the input in the label field-
    // If a cluster is already marked as isolated, it will not be used to train the classifier and it's automatically labelled as zero (as good point)
    for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
    {
      if ( (*clusters_data) [c_it].is_isolated)
        (*clusters_data) [c_it].features.label = 0;
      else
        (*clusters_data) [c_it].features.label =
          (double) getInputLabel (cloud_in, (*clusters_data) [c_it].indices, c_it + 1, clusters_data->size (), viewer);
    }

//         double labels[] = {2,0,2,1,0,0,0,0,0,1,0,0,0};
//         for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
//         {
//             (*clusters_data)[c_it].features.label = labels[c_it];
//         }

    // Close the viewer
    viewer->close();

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
    ml_svm_training.saveNormTrainingSet ("data_input_norm"); // Save clusters features normalized

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
initVisualizer (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, 
		const pcl::PointCloud<PointType>::Ptr cloud_in)
{
  // Setting the initial viewer parameters 
  viewer->initCameraParameters ();
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addCoordinateSystem (1000);
  viewer->camera_.view[0] = 0;
  viewer->camera_.view[1] = 0;
  viewer->camera_.view[2] = 1;
  viewer->camera_.pos[0] = 8000;
  viewer->camera_.pos[1] = 20000;
  viewer->camera_.pos[2] = 2500;
  viewer->updateCamera ();
  
  //pcl::visualization::PointCloudColorHandlerGenericField<PointType> rgb (global_data.cloud_octree, "intensity");
  viewer->addPointCloud<PointType>(global_data.cloud_octree, "cloud_in");
}

int
getInputLabel (const pcl::PointCloud<PointType>::Ptr cloud_in, pcl::IndicesPtr indices_, int i, int n_clusters_,
               boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{

  // Create a point cloud copy with the cluster info
  pcl::PointCloud<PointType>::Ptr cluster (new pcl::PointCloud<PointType>);
  pcl::copyPointCloud (*cloud_in, *indices_, *cluster);
  int stop = -1;
  bool delBuff = 0;

  // Get color handler for the cluster cloud
  pcl::visualization::PointCloudColorHandlerGenericField<PointType> rgb (cluster, "intensity");
  //int v2(0);
  //viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
  viewer->removePointCloud ("cloud"); // Clean viewer
  viewer->addPointCloud<PointType> (cluster, rgb, "cloud"); // Add the pointcloud
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud"); // Set viewer point size

  // Add text
  viewer->removeText3D ("user");
  viewer->addText ("Input the label for the displayed cluster: \n 0 : Good Points\n 1 : Ghost Points\n 2 : Vegetation Points", 50, 300, "user");
  
  // update the number of clusters
  viewer->removeText3D ("input");
  std::stringstream num;
  num << "cluster " << i << " of " << n_clusters_;
  viewer->addText (num.str ().data (), 100, 3, "input");

  // clean to reset the standard input of the viewer
  std::cin.clear();
  
  while (stop == 0)
  {
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    stop = 1;
    viewer->registerKeyboardCallback (deleteBuffer, (void*) &stop);
    viewer->spinOnce (500);
  }
  
  // clean to reset the standard input of the viewer
  std::cin.clear();
  
  // Spin the viewer until 0, 1 or 2 are pressed
  while (stop == -1)
  {
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    viewer->registerKeyboardCallback (keyboardEventOccurred, (void*) &stop);
    viewer->spinOnce (500);
  }
  
  // clean to reset the standard input of the viewer
  std::cin.clear();
  
  // Return the input value
  return stop;
}

void
keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* stop_void)
{
  int *stop;
  char *keyPressed = new char[50];

  // Copy the pressed key inside a var
  sprintf (keyPressed, "%c", event.getKeyCode ());
  stop = static_cast<int *> (stop_void);

  // Check if 0, 1 or 2 are pressed and return the value
  if (strpbrk ("012", keyPressed))
  {
    *stop = atoi (keyPressed);
    //std::cout << "found " << *stop << std::endl;
  }

  // Avoiding the window closing with 'q' and 'e'
  if (event.getKeyCode () == 'q' || event.getKeyCode () == 'e')
  {
    //std::cout << "quit " << keyPressed << std::endl;
    *stop = -1;
  }
}

void
deleteBuffer (const pcl::visualization::KeyboardEvent &event, void* stop_void)
{
  bool *stop;
  stop = static_cast<bool *> (stop_void);
  *stop = false;
}