#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree.h>

#include "regionGrowing.h"
#include "svm_wrapper.h"
#include "classification.h"

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
  std::vector<pcl::IndicesPtr> clusteredGoodIndices;
  std::vector<pcl::IndicesPtr> clusteredNoisyIndices;
  std::vector<pcl::IndicesPtr> clusteredTotalIndices;
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
    cerr << "  ex:    " << argv[0] << " noisy_points.pcd good_points.pcd 50 0.8 output.pcd\n" << endl;
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
    cout << "normals estimed for " << argv[1] << endl;

    rgB.setEpsAngle (atof (argv[4]));
    ne.setInputCloud (good_cloud);
    ne.setSearchMethod (tree);
    ne.setKSearch (atof (argv[3]));
    ne.compute (*good_normals);
    rgB.setNormals (good_normals);
    cout << "normals estimed for " << argv[2] << endl;
  }

  // Create the clusters
  cout << "Clustering noisy points...";

  rgA.setGrowingDistance (atof (argv[3]));

  rgA.cluster (&clusteredNoisyIndices);

  cout << "done. Found: " << clusteredNoisyIndices.size() << " clusters." << endl;

  cout << "Extracting point featuresA...";

  classification<PointType> featuresA (noise_cloud, clusteredNoisyIndices);

  cout << "done." << endl;

  cout << "Clustering good points...";

  rgB.setGrowingDistance (atof (argv[3]));

  rgB.cluster (&clusteredGoodIndices);

  cout << "done. Found: " << clusteredGoodIndices.size() << " clusters." << endl;

  // Extracting features from clusters
  cout << "Extracting point featuresB...";

  // classification<PointType> featuresA(noise_cloud, clusteredNoisyIndices);
  classification<PointType> featuresB (good_cloud, clusteredGoodIndices);

  cout << "done." << endl;

  // Creating a single cloud with labels
  total_cloud->operator += (*noise_cloud);

  total_cloud->operator += (*good_cloud);

  clusteredTotalIndices.insert (clusteredTotalIndices.end(), clusteredNoisyIndices.begin(), clusteredNoisyIndices.end());

  clusteredTotalIndices.insert (clusteredTotalIndices.end(), clusteredGoodIndices.begin(), clusteredGoodIndices.end());

  // Reprojecting the clusteredGoodIndices for the total cloud
  for (int i = 0; i < clusteredGoodIndices.size(); i++)
    for (int j = 0; j < clusteredGoodIndices[i]->size(); j++)
    {
      clusteredGoodIndices[i]->operator[] (j) = clusteredGoodIndices[i]->operator[] (j) + noise_cloud->size();
    }

  // Generating the vector for SVM training
  pcl::SvmTrain train;

  pcl::SVMParam param;

  param.C = 8192;

  param.gamma = 2;

  param.probability = 1;

  train.setParameters (param);

  cout << "Training the classifier...";

  for (int i = 0; i < featuresA.features.size(); i++)
    featuresA.features[i].label = new double (0); // 0 for noisy points

  for (int i = 0; i < featuresB.features.size(); i++)
    featuresB.features[i].label = new double (1.0); // 1 for good points

  train.setInputTrainingSet (featuresA.features);

  train.setInputTrainingSet (featuresB.features);

  cout << "done" << endl;

  train.setDebugMode (0);

  train.trainClassifier();

  train.saveProblemNorm ("theatrea");

  train.saveProblem ("theatreb");

  train.saveModel ("output.model");

  pcl::SvmClassify pred;

  pred.setInputModel (train.getOutputModel());

  pred.setInputTrainingSet (featuresA.features);

  pred.setInputTrainingSet (featuresB.features);

  pred.setProbabilityEstimates (1);

  pred.predictionTest();

  pcl::PointCloud<PointType>::Ptr buff_cloud (new pcl::PointCloud<PointType>);

  pcl::PointCloud<PointType>::Ptr output_cloud (new pcl::PointCloud<PointType>);

  std::vector< std::vector<double> > prediction;

  pred.getPrediction (prediction);

  for (int i = 0; i < clusteredTotalIndices.size(); i++)
  {
    pcl::copyPointCloud (*total_cloud, clusteredTotalIndices[i].operator*(), *buff_cloud);
    int j;

    for (j = 0; j < buff_cloud->size();j++)
      if (prediction[i][0] == 1)
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

  pcl::io::savePCDFileBinary (argv[5], *output_cloud);

  return 0;
}
