#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree.h>

#include "regionGrowing.h"
#include "SVM/svm_wrapper.h"
#include "extract_features.h"

using namespace std;
typedef pcl::PointXYZI PointType;

int main (int argc, char **argv)
{

  if (argc < 4)
  {
    cerr << "This program clusters the input cloud and predicts the presence of noises." << endl;
    cerr << "It requires output.model to load the classifier model." << endl;
    cerr << "usage: " << argv[0] << " [input.pcd/input.model] output.pcd clustering_ray" << endl;
    exit (0);
  }

  pcl::PointCloud<PointType>::Ptr input_cloud (new pcl::PointCloud<PointType>);

  pcl::PointCloud<PointType>::Ptr output_cloud (new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr buff_cloud (new pcl::PointCloud<PointType>);
  std::vector<pcl::IndicesPtr> clusteredIndices;
  pcl::RegionGrowing<PointType> rg;

  // Prediction
  pcl::SVMClassify pred;
  pred.loadClassifierModel ("output.model");

  size_t found;
  string filename;
  filename.assign (argv[1]);
  //cout << filename << endl;
  found = filename.find ("model", 0);

  if (found == string::npos)
  {

    // Load the clouds
    if (pcl::io::loadPCDFile (argv[1], *input_cloud))
      return 0;

    // Make the cloud unordered to avoid knearest search conflicts
    input_cloud->width = input_cloud->size();

    input_cloud->height = 1;

    input_cloud->is_dense = 0;


    // Create the clusters
    cout << "Clustering points...";

    rg.setInputCloud (input_cloud);

    rg.setGrowingDistance (atof (argv[3]));

    rg.cluster (&clusteredIndices);

    cout << "done." << endl;

    // Extracting features from clusters
    cout << "Extracting point features...";

    ExtractFeatures<PointType> features (input_cloud, clusteredIndices);
    cout << "done." << endl;

    pred.setInputTrainingSet (features.features);
    string out_name;

    out_name.assign (argv[1]);
    out_name.append (".model");
    //pred.saveProblem(out_name.data());
    pred.saveClassProblem ("ab");
    pred.saveNormClassProblem ("ba");
    cout << "Computing classification..." ;

    pred.classification();
    cout << "done." << endl;

    std::vector< std::vector<double> > prediction;
    pred.getClassificationResult (prediction);

    // Save the results
    for (int i = 0; i < clusteredIndices.size(); i++)
    {
      pcl::copyPointCloud (*input_cloud, clusteredIndices[i].operator*(), *buff_cloud);
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

    pcl::io::savePCDFileBinary (argv[2], *output_cloud);

    return 0;

  }
  else
  {
    pcl::SVMData test;
    test.SV.resize (2);
    test.SV[0].idx = 2;
    test.SV[0].value = 35;
    test.SV[1].idx = 5;
    test.SV[1].value = 0.35;

    pred.setProbabilityEstimates (1);
    pred.loadClassProblem (argv[1]);
    cout << "Computing classification..." ;
    pred.classification();
    pred.classificationTest();
    pred.saveClassificationResult ("prediction");
    std::vector<double> out;
    out =  pred.classification (test);
    cout << "Single prediction: ";

    for (int i = 0 ; i < out.size(); i++)
      cout << out[i] << " ";

    cout << "\ndone." << endl;

    //cout << "va "<< pred.prob_.l << endl;

    pred.saveClassProblem ("ab");
    pred.saveNormClassProblem ("ba");
    pred.saveClassifierModel ("output.model.2");

    return 0;
  }
}