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

// TODO The function checks for the existence of an existing model. If it fails to load the model, it start a new training procedure of the classifier.

/** \brief The machine learning classifier, results are stored in the ClusterData structs.
  * \param[in] cloud_in A pointer to the input point cloud.
  * \param[in] global_data A struct holding information on the full point cloud and global input parameters.
  * \param[in] model_filename Filename of the classifier model.
  * \param[in/out] clusters_data An array of information holders for each cluster
  */
void
applyObjectClassification (const pcl::PointCloud<PointType>::Ptr cloud_in,
                           GlobalData global_data,
                           boost::shared_ptr<std::vector<ClusterData> > &clusters_data,
			   const char *model_filename  )
{
  // Set up the machine learnin class
  pcl::SvmTrain ml_svm_training; // To train the classifier
  pcl::SvmClassify ml_svm_classify; // To classify
  
  // If the input model_filename exists, it starts the classification. 
  // Otherwise it starts a new machine learning training.
    if ( ml_svm_classify.loadModel(model_filename)) {
        pcl::console::print_highlight (stderr, "Loaded ");
        pcl::console::print_value (stderr, "%s ", model_filename);
	ml_svm_classify.setInputTrainingSet(global_data.features);
	ml_svm_classify.saveProblem("normal");
	ml_svm_classify.saveProblemNorm("normalized");
	ml_svm_classify.predict();
    } else {
      ml_svm_training.setInputTrainingSet( global_data.features );
      ml_svm_training.saveProblem("normal");
      ml_svm_training.saveProblemNorm("normalized");
      ml_svm_training.train();
      ml_svm_training.saveModel(model_filename);
    }
  
//   // Passthrough example: every cluster that has features[0] > 0.5 will be classified as ghost
//   for (size_t c_it = 0; c_it < clusters_data->size (); ++c_it)
//     if ((*clusters_data)[c_it].features[0] > 0.5)
//       (*clusters_data)[c_it].is_ghost = true;
}
