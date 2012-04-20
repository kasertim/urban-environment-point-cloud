#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree.h>

#include "regionGrowing.h"
#include "SVM/svm_wrapper.h"
#include "extract_features.h"

#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d.h>

using namespace std;
typedef pcl::PointXYZI PointType;



int main (int argc, char **argv)
{
  pcl::PointCloud<PointType>::Ptr noise_cloud (new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr good_cloud (new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr total_cloud (new pcl::PointCloud<PointType>);
  pcl::PointCloud<pcl::Normal>::Ptr noise_normals (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::Normal>::Ptr good_normals (new pcl::PointCloud<pcl::Normal>);
  std::vector<pcl::IndicesPtr> clustered_good_indices;
  std::vector<pcl::IndicesPtr> clustered_noisy_indices;
  std::vector<pcl::IndicesPtr> clustered_total_indices;
  pcl::RegionGrowing<PointType> rgA, rgB;

  if (argc < 6)
  {
    cerr << "\nThis program computes the classifier starting from two input clouds having x,y,z and intensity info: " << endl;
    cerr << "\n  usage: " << argv[0] << " noise_cloud.pcd good_cloud.pcd radius angle output.pcd" << endl;
    cerr << "  - noise_cloud: from which it extracts the segments labelled as noisy" << endl;
    cerr << "  - good_cloud: from which it extracts the segments labelled as good" << endl;
    cerr << "  - clustering_radius: radius used for neighbour search" << endl;
    cerr << "  - normals_offset: max angle between two point normals [radiants]" << endl;
    cerr << "  - output: outputs the cloud showing the classification results\n" << endl;
    cerr << "  ex:    " << argv[0] << " cropped_ghosts.pcd cropped_leaves.pcd 50 0 output.pcd\n" << endl;
    exit (0);
  }

  // Load the clouds
  if (pcl::io::loadPCDFile (argv[1], *noise_cloud))
    return 0;

  if (pcl::io::loadPCDFile (argv[2], *good_cloud))
    return 0;

  rgA.setInputCloud (noise_cloud);

  rgB.setInputCloud (good_cloud);

  // Extracting normals
  if (atof (argv[4]) > 0)
  {
    rgA.setEpsAngle (atof (argv[4]));
    pcl::NormalEstimation<PointType, pcl::Normal> ne;
    ne.setInputCloud (noise_cloud);
    pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> (false));
    ne.setSearchMethod (tree);
    ne.setKSearch (atof (argv[3]));
    ne.compute (*noise_normals);
    rgA.setNormals (noise_normals);
    cout << "normals calculated for " << argv[1] << endl;

    rgB.setEpsAngle (atof (argv[4]));
    ne.setInputCloud (good_cloud);
    ne.setSearchMethod (tree);
    ne.setKSearch (atof (argv[3]));
    ne.compute (*good_normals);
    rgB.setNormals (good_normals);
    cout << "Normals calculated for " << argv[2] << endl;
  }

  // Create the clusters
  cout << "Clustering noisy points...";

  rgA.setGrowingDistance (atof (argv[3]));
  rgA.cluster (&clustered_noisy_indices);

  cout << "done. Found: " << clustered_noisy_indices.size() << " clusters." << endl;
  cout << "Extracting point features_a...";

  ExtractFeatures<PointType> features_a (noise_cloud, clustered_noisy_indices);

  cout << "done." << endl;
  cout << "Clustering good points...";

  rgB.setGrowingDistance (atof (argv[3]));
  rgB.cluster (&clustered_good_indices);

  cout << "done. Found: " << clustered_good_indices.size() << " clusters." << endl;

  // Extracting features from clusters
  cout << "Extracting point features_b...";

  ExtractFeatures<PointType> features_b (good_cloud, clustered_good_indices);

  cout << "done." << endl;

  // Creating a single cloud with labels
  total_cloud->operator += (*noise_cloud);
  total_cloud->operator += (*good_cloud);

  clustered_total_indices.insert (clustered_total_indices.end(), clustered_noisy_indices.begin(), clustered_noisy_indices.end());
  clustered_total_indices.insert (clustered_total_indices.end(), clustered_good_indices.begin(), clustered_good_indices.end());

  // Reprojecting the clustered_good_indices for the total cloud
  for (int i = 0; i < clustered_good_indices.size(); i++)
    for (int j = 0; j < clustered_good_indices[i]->size(); j++)
    {
      clustered_good_indices[i]->operator[] (j) = clustered_good_indices[i]->operator[] (j) + noise_cloud->size();
    }

  // Generating the class for SVM training
  pcl::SVMTrain train;
  
  // Set parameters
  pcl::SVMParam param;
  param.probability = 1; // To do probability estimation
  param.C = 8192;
  param.gamma = 2;
  train.setParameters (param);

  cout << "Training the classifier...";

  for (int i = 0; i < features_a.features.size(); i++)
    features_a.features[i].label = 0.0f; // 0 for noisy points

  for (int i = 0; i < features_b.features.size(); i++)
    features_b.features[i].label = 1.0f; // 1 for good points

  train.setInputTrainingSet (features_a.features); // Set input features_a
  train.setInputTrainingSet (features_b.features); // Append features_b to already loaded features_a

  cout << "done" << endl;

  train.setDebugMode (1);
  train.trainClassifier(); // run the training
  
  // Save output files
  train.saveNormTrainingSet ("normalized_training_set");
  train.saveTrainingSet ("training_set");
  train.saveClassifierModel ("classifier_model");
 
  // Define the classification class
  pcl::SVMClassify pred;
  pred.setClassifierModel (train.getClassifierModel()); // Copy the classifier model from the training step
  pred.setInputTrainingSet (features_a.features); // Set input features_a
  pred.setInputTrainingSet (features_b.features); // Append features_b to already loaded features_a
  pred.setProbabilityEstimates (1); // Probability estimation
  pred.classificationTest(); // run the classification test to double-check the classifier

  
  pcl::PointCloud<PointType>::Ptr buff_cloud (new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr output_cloud (new pcl::PointCloud<PointType>);
  std::vector< std::vector<double> > classification_out;

  pred.getClassificationResult (classification_out); // get the classification result

  // Create a cloud highlighting the classification result
  for (int i = 0; i < clustered_total_indices.size(); i++)
  {
    pcl::copyPointCloud (*total_cloud, clustered_total_indices[i].operator*(), *buff_cloud);
    int j;

    for (j = 0; j < buff_cloud->size();j++)
      if (classification_out[i][0] == 1)
      {
        buff_cloud->points[j].intensity = 255;
      }
      else
      {
        buff_cloud->points[j].intensity = 0;
      }

    //cout << buff_cloud->points[j-1].intensity << endl;
    output_cloud->operator+= (*buff_cloud);

    buff_cloud->clear();
  }

  // Save the classified cloud
  pcl::io::savePCDFileBinary (argv[5], *output_cloud);

  return 0;
}
