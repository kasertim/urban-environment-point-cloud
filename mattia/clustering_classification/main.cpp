#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree.h>

#include "regionGrowing.h"
#include "svm-train.h"
#include "classification.h"

#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d.h>

using namespace std;
typedef pcl::PointXYZI PointType;

int main(int argc, char **argv) {
    pcl::PointCloud<PointType>::Ptr noise_cloud (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr good_cloud (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr total_cloud (new pcl::PointCloud<PointType>);
    std::vector<pcl::IndicesPtr> clusteredGoodIndices;
    std::vector<pcl::IndicesPtr> clusteredNoisyIndices;
    std::vector<pcl::IndicesPtr> clusteredTotalIndices;
    pcl::RegionGrowing<PointType> rgA, rgB;

    if (argc < 5)
    {
        cerr << "\nThis program computes the classifier starting from two input clouds having x,y,z and intensity info: " << endl;
        cerr << "\n  usage: " << argv[0] << " noise_cloud.pcd good_cloud.pcd radius output.pcd" << endl;
	cerr << "  - noise_cloud: from which it extracts the segments labelled as noisy" << endl;
        cerr << "  - good_cloud: from which it extracts the segments labelled as good" << endl;
	cerr << "  - clustering_radius: radius used for neighbour search" << endl;
	cerr << "  - output: outputs the cloud showing the classification results\n" << endl;
	cerr << "  ex:    "<< argv[0] << " noisy_points.pcd good_points.pcd 50 output.pcd\n" << endl;
        exit(0);
    }

    // Load the clouds
    if (pcl::io::loadPCDFile (argv[1], *noise_cloud))
        return 0;
    if (pcl::io::loadPCDFile (argv[2], *good_cloud))
        return 0;

//     pcl::PointCloud<pcl::VFHSignature308>::Ptr vfh_temp (new pcl::PointCloud<pcl::VFHSignature308>);
//     std::vector<pcl::PointCloud<pcl::VFHSignature308>::Ptr> vfh_ptrs_;
//     pcl::VFHEstimation<PointType, pcl::Normal, pcl::VFHSignature308> vfher_;
//     typename pcl::search::KdTree<PointType>::Ptr tree (new typename pcl::search::KdTree<PointType> ());
//     pcl::PointCloud<pcl::Normal>::Ptr normals_all (new pcl::PointCloud<pcl::Normal>);
//     pcl::NormalEstimation<PointType, pcl::Normal> ne;
//     
//     ne.setInputCloud (good_cloud);
//     ne.setSearchMethod (tree);
//     ne.setKSearch(20);
//     ne.compute (*normals_all);
// 
//     vfher_.setInputCloud (good_cloud);
//     vfher_.setInputNormals (normals_all);
//     vfher_.setSearchMethod (tree);
//     //vfher_.setIndices( clusters_[i] );
// //    vfher_.compute (*vfh_temp);
//     vfh_temp.reset(new pcl::PointCloud<pcl::VFHSignature308>);
//     vfh_temp->clear();
//     vfh_temp->points.clear();
//     vfh_ptrs_.push_back (vfh_temp);
//     
//     cout << "size " << vfh_ptrs_.size() << endl;
//     cout << " and1 " << vfh_ptrs_[0]->size() << endl;
//     cout<< " and2 "<< vfh_ptrs_[0]->points[0].histogram[0]<< endl;
	    
    // Create the clusters
    cout << "Clustering noisy points...";
    rgA.setInputCloud (noise_cloud);
    rgA.setGrowingDistance(atof(argv[3]));
    rgA.cluster (&clusteredNoisyIndices);
    cout << "done. Found: " << clusteredNoisyIndices.size()<< " clusters." << endl;
	
    cout << "Clustering good points...";
    rgB.setInputCloud (good_cloud);
    rgB.setGrowingDistance(atof(argv[3]));
    rgB.cluster (&clusteredGoodIndices);
    cout << "done. Found: " << clusteredGoodIndices.size()<< " clusters."<< endl;

    // Extracting features from clusters
    cout << "Extracting point features...";
    classification<PointType> featuresA(noise_cloud, clusteredNoisyIndices);
    classification<PointType> featuresB(good_cloud, clusteredGoodIndices);
    cout << "done." << endl;

    // Creating a single cloud with labels
    total_cloud->operator += (*noise_cloud);
    total_cloud->operator += (*good_cloud);
    
    clusteredTotalIndices.insert(clusteredTotalIndices.end(), clusteredNoisyIndices.begin(), clusteredNoisyIndices.end());
    clusteredTotalIndices.insert(clusteredTotalIndices.end(), clusteredGoodIndices.begin(), clusteredGoodIndices.end());

    // Reprojecting the clusteredGoodIndices for the total cloud
    for (int i=0; i< clusteredGoodIndices.size(); i++)
        for (int j=0; j < clusteredGoodIndices[i]->size(); j++) {
            clusteredGoodIndices[i]->operator[](j) = clusteredGoodIndices[i]->operator[](j) + noise_cloud->size();
        }


    
    // Generating the vector for SVM training
    SvmTrain train;
    
    train.nFeatures= 5 + 308; // n of features for a point
    train.prob.l = clusteredNoisyIndices.size() + clusteredGoodIndices.size(); // n of elements/points
    train.prob.y = Malloc(double,train.prob.l);
    train.prob.x = Malloc(struct svm_node *,train.prob.l);
    train.scaling.obj = Malloc(struct svm_node,train.nFeatures+1);
    train.param.C=8192;//32768;
    train.param.gamma=2;//8;
//     train.cross_validation=1;
//     train.nr_fold = 8; // is how many sets to split your input data
    
    // saving the scaling factors
    cout << "Scaling the features...";
    train.scaling.obj[0].index=0;
    if ( *( std::max_element( featuresA.cardinality_.begin(), featuresA.cardinality_.end() ) ) >
         *( std::max_element( featuresB.cardinality_.begin(), featuresB.cardinality_.end() ) ) )
        train.scaling.obj[0].value=*( std::max_element( featuresA.cardinality_.begin(), featuresA.cardinality_.end() ) );
    else
        train.scaling.obj[0].value=*( std::max_element( featuresB.cardinality_.begin(), featuresB.cardinality_.end() ) );
    
    train.scaling.obj[1].index=1;
    if ( *( std::max_element( featuresA.intensity_.begin(), featuresA.intensity_.end() ) ) >
         *( std::max_element( featuresB.intensity_.begin(), featuresB.intensity_.end() ) ) )
        train.scaling.obj[1].value=*( std::max_element( featuresA.intensity_.begin(), featuresA.intensity_.end() ) );
    else
        train.scaling.obj[1].value=*( std::max_element( featuresB.intensity_.begin(), featuresB.intensity_.end() ) );
    
    train.scaling.obj[2].index=2;
    if ( *( std::max_element( featuresA.norm_std_dev_.begin(), featuresA.norm_std_dev_.end() ) ) >
         *( std::max_element( featuresB.norm_std_dev_.begin(), featuresB.norm_std_dev_.end() ) ) )
        train.scaling.obj[2].value=*( std::max_element( featuresA.norm_std_dev_.begin(), featuresA.norm_std_dev_.end() ) );
    else
        train.scaling.obj[2].value=*( std::max_element( featuresB.norm_std_dev_.begin(), featuresB.norm_std_dev_.end() ) );
    
    train.scaling.obj[3].index=3;
    if ( *( std::max_element( featuresA.curv_std_dev_.begin(), featuresA.curv_std_dev_.end() ) ) >
         *( std::max_element( featuresB.curv_std_dev_.begin(), featuresB.curv_std_dev_.end() ) ) )
        train.scaling.obj[3].value=*( std::max_element( featuresA.curv_std_dev_.begin(), featuresA.curv_std_dev_.end() ) );
    else
        train.scaling.obj[3].value=*( std::max_element( featuresB.curv_std_dev_.begin(), featuresB.curv_std_dev_.end() ) );
    
    train.scaling.obj[4].index=4;
    if ( *( std::max_element( featuresA.eigModule_.begin(), featuresA.eigModule_.end() ) ) >
         *( std::max_element( featuresB.eigModule_.begin(), featuresB.eigModule_.end() ) ) )
        train.scaling.obj[4].value=*( std::max_element( featuresA.eigModule_.begin(), featuresA.eigModule_.end() ) );
    else
        train.scaling.obj[4].value=*( std::max_element( featuresB.eigModule_.begin(), featuresB.eigModule_.end() ) );
    
    // Finding the maximum values for 308 elements
    for(int vfh_n=0; vfh_n < 308; vfh_n++){
      train.scaling.obj[4+1+vfh_n].index=4+1+vfh_n;
      
      float sumA=0.0f, sumB=0.0f;
      int a_num=0, b_num=0;
      for(int jj=0; jj < featuresA.vfh_ptrs_.size(); jj++)
	if(featuresA.vfh_ptrs_[jj]->size() > 0){
	  sumA += featuresA.vfh_ptrs_[jj]->points[0].histogram[vfh_n];
	  a_num++;
	}
      sumA = sumA / a_num;

      for(int jj=0; jj < featuresB.vfh_ptrs_.size(); jj++)
	if(featuresB.vfh_ptrs_[jj]->size() > 0){
	  sumB += featuresB.vfh_ptrs_[jj]->points[0].histogram[vfh_n];
	  b_num++;
	}
      sumB = sumB / b_num;
      
      if(sumA > sumB)
	train.scaling.obj[4+1+vfh_n].value = sumA;
      else
	train.scaling.obj[4+1+vfh_n].value = sumB;
    }
    
    train.scaling.obj[train.nFeatures + 1].index=-1;
    cout << "done." << endl;
    // display the maximum
// 	cout << train.scaling.obj[0].value << endl;
// 	cout << train.scaling.obj[1].value << endl;
// 	cout << train.scaling.obj[2].value << endl;
// 	cout << train.scaling.obj[3].value << endl;
// 	cout << train.scaling.obj[4].value << endl;
	
    cout << "Training the classifier...";
    // Fill the training set with noisy data
    for (int i=0;i<clusteredNoisyIndices.size();i++)
    {
        train.prob.y[i] = 0; // label 0 for noise, 1 for good
        train.prob.x[i] = Malloc(struct svm_node,train.nFeatures+1);
        int j=0;

        if ( std::isfinite(featuresA.cardinality_[i]) )
        {
            train.prob.x[i][j].index = 0;
            train.prob.x[i][j].value = featuresA.cardinality_[i] / train.scaling.obj[0].value;
            j++;
        }

        if ( std::isfinite(featuresA.intensity_[i]) )
        {
            train.prob.x[i][j].index = 1;
            train.prob.x[i][j].value = featuresA.intensity_[i] / train.scaling.obj[1].value;
            j++;
        }

        if ( std::isfinite(featuresA.norm_std_dev_[i]) )
        {
            train.prob.x[i][j].index = 2;
            train.prob.x[i][j].value = featuresA.norm_std_dev_[i] / train.scaling.obj[2].value;
            j++;
        }

        if ( std::isfinite(featuresA.curv_std_dev_[i]) ) {
            train.prob.x[i][j].index = 3;
            train.prob.x[i][j].value = featuresA.curv_std_dev_[i] / train.scaling.obj[3].value;
            j++;
        }

        if ( std::isfinite(featuresA.eigModule_[i]) )
        {
            train.prob.x[i][j].index = 4;
            train.prob.x[i][j].value = featuresA.eigModule_[i] / train.scaling.obj[4].value;
            j++;
        }

        if ( featuresA.vfh_ptrs_[i]->size() > 0 )
            for (int vfh_n=0; vfh_n < 308; vfh_n++) {
                if ( std::isfinite(featuresA.vfh_ptrs_[i]->points[0].histogram[vfh_n]) ) {
                    train.prob.x[i][j].index = 4+vfh_n+1;
                    train.prob.x[i][j].value =
                        featuresA.vfh_ptrs_[i]->points[0].histogram[vfh_n] / train.scaling.obj[4+vfh_n+1].value;
                    j++;
                }
            }

        train.prob.x[i][j].index = -1; // set last element of a sample
    }

    // Fill the training set with good data
    for (int i=clusteredNoisyIndices.size(); i < clusteredNoisyIndices.size()+clusteredGoodIndices.size(); i++)
    {
        train.prob.y[i] = 1; // label 0 for noise, 1 for good
        train.prob.x[i] = Malloc(struct svm_node,train.nFeatures+1);
        int j=0;

        if ( std::isfinite(featuresB.cardinality_[i-clusteredNoisyIndices.size()]) )
        {
            train.prob.x[i][j].index = 0;
            train.prob.x[i][j].value = featuresB.cardinality_[i-clusteredNoisyIndices.size()] / train.scaling.obj[0].value;
            j++;
        }

        if ( std::isfinite(featuresB.intensity_[i-clusteredNoisyIndices.size()]) )
        {
            train.prob.x[i][j].index = 1;
            train.prob.x[i][j].value = featuresB.intensity_[i-clusteredNoisyIndices.size()] / train.scaling.obj[1].value;
            j++;
        }

        if ( std::isfinite(featuresB.norm_std_dev_[i-clusteredNoisyIndices.size()]) )
        {
            train.prob.x[i][j].index = 2;
            train.prob.x[i][j].value = featuresB.norm_std_dev_[i-clusteredNoisyIndices.size()] / train.scaling.obj[2].value;
            j++;
        }

        if ( std::isfinite(featuresB.curv_std_dev_[i-clusteredNoisyIndices.size()]) ) {
            train.prob.x[i][j].index = 3;
            train.prob.x[i][j].value = featuresB.curv_std_dev_[i-clusteredNoisyIndices.size()] / train.scaling.obj[3].value;
            j++;
        }

        if ( std::isfinite(featuresB.eigModule_[i-clusteredNoisyIndices.size()]) )
        {
            train.prob.x[i][j].index = 4;
            train.prob.x[i][j].value = featuresB.eigModule_[i-clusteredNoisyIndices.size()] / train.scaling.obj[4].value;
            j++;
        }

        if ( featuresB.vfh_ptrs_[i]->size() > 0 )
            for (int vfh_n=0; vfh_n < 308; vfh_n++) {
                if ( std::isfinite(featuresB.vfh_ptrs_[i]->points[0].histogram[vfh_n]) ) {
                    train.prob.x[i][j].index = 4+vfh_n+1;
                    train.prob.x[i][j].value =
                        featuresB.vfh_ptrs_[i]->points[0].histogram[vfh_n] / train.scaling.obj[4+vfh_n+1].value;
                    j++;
                }
            }
        
        train.prob.x[i][j].index = -1; // set last element of a sample
    }
    cout << "done" << endl;
    train.saveProblem("theatre");

    train.execute();

    SvmPredict pred;
    pred.model = train.model;
    pred.input = train.prob;
    pred.prediction_test();

    pcl::PointCloud<PointType>::Ptr buff_cloud (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr output_cloud (new pcl::PointCloud<PointType>);

    for (int i=0; i<clusteredTotalIndices.size(); i++) {
        pcl::copyPointCloud(*total_cloud, clusteredTotalIndices[i].operator*(), *buff_cloud);
	int j;
        for (j=0; j<buff_cloud->size();j++)
            if (pred.prediction[i]==1){
                buff_cloud->points[j].intensity = 255;
	    }
            else{
                buff_cloud->points[j].intensity = 0;
		 
	    }
	//cout << buff_cloud->points[j-1].intensity << endl;
        output_cloud->operator+=(*buff_cloud);
        buff_cloud->clear();
    }

    pcl::io::savePCDFileBinary(argv[4], *output_cloud);
    return 0;
}
