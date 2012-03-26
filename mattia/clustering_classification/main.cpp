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
    pcl::PointCloud<PointType>::Ptr noise_cloud (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr good_cloud (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr total_cloud (new pcl::PointCloud<PointType>);
    std::vector<pcl::IndicesPtr> clusteredGoodIndices;
    std::vector<pcl::IndicesPtr> clusteredNoisyIndices;
    pcl::RegionGrowing<PointType> rgA, rgB;

    if (argc < 4)
    {
        cerr << "This program computes the classifier starting from two input clouds having x,y,z and intensity info: " << endl;
        cerr << "  - noise_cloud: from which it extracts the segments labelled as noisy" << endl;
        cerr << "  - good_cloud: from which it extracts the segments labelled as good\n" << endl;
        cerr << "usage: " << argv[0] << " noise_cloud.pcd good_cloud.pcd clustering_ray" << endl;
        exit(0);
    }

    // Load the clouds
    if (pcl::io::loadPCDFile (argv[1], *noise_cloud))
        return 0;
    if (pcl::io::loadPCDFile (argv[2], *good_cloud))
        return 0;


    // Create the clusters
    cout << "Clustering noisy points...";
    rgA.setInputCloud (noise_cloud);
    rgA.setGrowingDistance(atof(argv[3]));
    rgA.cluster (&clusteredNoisyIndices);
    cout << "done." << endl;
    cout << "noisy clusters? " <<clusteredNoisyIndices.size()<<endl;

    cout << "Clustering good points...";
    rgB.setInputCloud (good_cloud);
    rgB.setGrowingDistance(atof(argv[3]));
    rgB.cluster (&clusteredGoodIndices);
    cout << "done." << endl;
    cout << "good clusters? " <<clusteredGoodIndices.size()<<endl;

    // Extracting features from clusters
    cout << "Extracting point features...";
    classification<PointType> featuresA(noise_cloud, clusteredNoisyIndices);
    classification<PointType> featuresB(good_cloud, clusteredGoodIndices);
    cout << "done." << endl;

    // Creating a single cloud with labels
    total_cloud->operator += (*noise_cloud);
    total_cloud->operator += (*good_cloud);

    // Reprojecting the clusteredGoodIndices for the total cloud
    for (int i=0; i< clusteredGoodIndices.size(); i++)
        for (int j=0; j < clusteredGoodIndices[i]->size(); j++) {
            clusteredGoodIndices[i]->operator[](j) = clusteredGoodIndices[i]->operator[](j) + noise_cloud->size();
        }

//     // Test to check if the reprojectione went well
//     for (int i=0; i<10; i++) {
//         int rand_n = rand()%2001;
//         cout << "\nTotal cloud test." << endl;
//         cout << "Random point " << rand_n << endl;
//         cout <<"1 "<<total_cloud->points[ clusteredNoisyIndices[rand_n]->operator[](0) ].x << " "
//              << total_cloud->points[ clusteredNoisyIndices[rand_n]->operator[](0) ].y << " "
//              << total_cloud->points[ clusteredNoisyIndices[rand_n]->operator[](0) ].z << endl;
//         cout <<"1 "<<noise_cloud->points[ clusteredNoisyIndices[rand_n]->operator[](0) ].x << " "
//              << noise_cloud->points[ clusteredNoisyIndices[rand_n]->operator[](0) ].y << " "
//              << noise_cloud->points[ clusteredNoisyIndices[rand_n]->operator[](0) ].z << endl;
//
//         cout  << "2 "<<total_cloud->points[ clusteredGoodIndices[rand_n]->operator[](0)].x << " "
//              << total_cloud->points[ clusteredGoodIndices[rand_n]->operator[](0) ].y << " "
//              << total_cloud->points[ clusteredGoodIndices[rand_n]->operator[](0) ].z << endl;
//         cout << "2 "<<good_cloud->points[ clusteredGoodIndices[rand_n]->operator[](0) - noise_cloud->size()].x << " "
//              << good_cloud->points[ clusteredGoodIndices[rand_n]->operator[](0) - noise_cloud->size()].y << " "
//              << good_cloud->points[ clusteredGoodIndices[rand_n]->operator[](0) - noise_cloud->size()].z << endl;
//         cout << endl;
//     }
    
    // Generating the vector for SVM training
    SvmTrain train;

    train.nFeatures=5; // n of features for a point
    train.prob.l = clusteredNoisyIndices.size() + clusteredGoodIndices.size(); // n of elements/points
    train.prob.y = Malloc(double,train.prob.l);
    train.prob.x = Malloc(struct svm_node *,train.prob.l);
    //train.cross_validation=1;
    //train.nr_fold = 4; // is how many sets to split your input data
    cout << "Training the classifier...";
    // Fill the training set with noisy data
    for (int i=0;i<clusteredNoisyIndices.size();i++)
    {
        train.prob.y[i] = 0; // label 0 for noise, 1 for good
        train.prob.x[i] = Malloc(struct svm_node,train.nFeatures+1);

        train.prob.x[i][0].index = 0;
        train.prob.x[i][0].value = featuresA.cardinality_[i];

        train.prob.x[i][1].index = 1;
        train.prob.x[i][1].value = featuresA.intensity_[i];

        train.prob.x[i][2].index = 2;
        train.prob.x[i][2].value = featuresA.norm_std_dev_[i];

        train.prob.x[i][3].index = 3;
        train.prob.x[i][3].value = featuresA.curv_std_dev_[i];

        train.prob.x[i][4].index = 4;
        train.prob.x[i][4].value = featuresA.eigModule_[i];

        train.prob.x[i][5].index = -1; // set last element of a sample
    }

    // Fill the training set with good data
    for (int i=clusteredNoisyIndices.size(); i < clusteredNoisyIndices.size()+clusteredGoodIndices.size(); i++)
    {
        train.prob.y[i] = 1; // label 0 for noise, 1 for good
        train.prob.x[i] = Malloc(struct svm_node,train.nFeatures+1);

        train.prob.x[i][0].index = 0;
        train.prob.x[i][0].value = featuresB.cardinality_[i-clusteredNoisyIndices.size()];

        train.prob.x[i][1].index = 1;
        train.prob.x[i][1].value = featuresB.intensity_[i-clusteredNoisyIndices.size()];

        train.prob.x[i][2].index = 2;
        train.prob.x[i][2].value = featuresB.norm_std_dev_[i-clusteredNoisyIndices.size()];

        train.prob.x[i][3].index = 3;
        train.prob.x[i][3].value = featuresB.curv_std_dev_[i-clusteredNoisyIndices.size()];

        train.prob.x[i][4].index = 4;
        train.prob.x[i][4].value = featuresB.eigModule_[i-clusteredNoisyIndices.size()];

        train.prob.x[i][5].index = -1; // set last element of a sample
    }
    cout << "done" << endl;
    // Train the classifier
    train.execute();

    return 0;
}
