#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree.h>

#include "regionGrowing.h"
#include "svm-train.h"
#include "classification.h"

using namespace std;
typedef pcl::PointXYZI PointType;

int main(int argc, char **argv) {

    if (argc < 4)
    {
        cerr << "This program clusters the input cloud and predicts the presence of noises." << endl;
        cerr << "It requires output.model to load the classifier model." << endl;
        cerr << "usage: " << argv[0] << " input.pcd output.pcd clustering_ray" << endl;
	exit(0);
    }

    pcl::PointCloud<PointType>::Ptr input_cloud (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr output_cloud (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr buff_cloud (new pcl::PointCloud<PointType>);
    std::vector<pcl::IndicesPtr> clusteredIndices;
    pcl::RegionGrowing<PointType> rg;

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
    rg.setGrowingDistance(atof(argv[3]));
    rg.cluster (&clusteredIndices);
    cout << "done." << endl;

    // Extracting features from clusters
    cout << "Extracting point features...";
    classification<PointType> features(input_cloud, clusteredIndices);
    cout << "done." << endl;

    // Prediction
    int nFeatures = 5;
    SvmPredict pred;
    pred.loadModel("output.model");
    pred.input.l = clusteredIndices.size(); // n of elements/points
    pred.input.y = Malloc(double,pred.input.l);
    pred.input.x = Malloc(struct svm_node *,pred.input.l);

    for (int i=0;i<clusteredIndices.size();i++)
    {
        pred.input.y[i] = 0; // label 0 for noise, 1 for good
        pred.input.x[i] = Malloc(struct svm_node,nFeatures+1);
        int j=0;

        if ( std::isfinite(features.cardinality_[i]) )
        {
            pred.input.x[i][j].index = 0;
            pred.input.x[i][j].value = features.cardinality_[i];
            j++;
        }

        if ( std::isfinite(features.intensity_[i]) )
        {
            pred.input.x[i][j].index = 1;
            pred.input.x[i][j].value = features.intensity_[i];
            j++;
        }

        if ( std::isfinite(features.norm_std_dev_[i]) )
        {
            pred.input.x[i][j].index = 2;
            pred.input.x[i][j].value = features.norm_std_dev_[i];
            j++;
        }

        if ( std::isfinite(features.curv_std_dev_[i]) ) {
            pred.input.x[i][j].index = 3;
            pred.input.x[i][j].value = features.curv_std_dev_[i];
            j++;
        }

        if ( std::isfinite(features.eigModule_[i]) )
        {
            pred.input.x[i][j].index = 4;
            pred.input.x[i][j].value = features.eigModule_[i];
            j++;
        }

        pred.input.x[i][j].index = -1; // set last element of a sample
    }

    cout << "Computing classification..." ;
    pred.predict();
    cout << "done." << endl;

    // Save the results
    for (int i=0; i<clusteredIndices.size(); i++) {
        pcl::copyPointCloud(*input_cloud, clusteredIndices[i].operator*(), *buff_cloud);
        int j;
        for (j=0; j<buff_cloud->size();j++)
            if (pred.prediction[i]==1) {
                buff_cloud->points[j].intensity = 255;
            }
            else {
                buff_cloud->points[j].intensity = 0;

            }
        //cout << buff_cloud->points[j-1].intensity << endl;
        output_cloud->operator+=(*buff_cloud);
        buff_cloud->clear();
    }

    pcl::io::savePCDFileBinary(argv[2], *output_cloud);
    return 0;
}