#include <boost/graph/graph_concepts.hpp>
#include <pcl/point_cloud.h>
#include <pcl/PointIndices.h>
#include <pcl/console/print.h>
#include <pcl/pcl_base.h>
#include "pcl/search/pcl_search.h"
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/common/pca.h>

#include "svm_wrapper.h"

// I added these
//#include <pcl/common/common.h>
#include <pcl/features/vfh.h>

#include <Eigen/Dense>
using namespace Eigen;

#define MAX_NORNALIZE 1//50000000.0

inline float module(float a) {
    if (a>0)
        return a;
    else
        return -a;
}

template <typename PointT>
class classification
{

public:
    typedef pcl::PointCloud<PointT> PointCloud;
    typedef typename PointCloud::Ptr PointCloudPtr;
    typedef typename PointCloud::ConstPtr PointCloudConstPtr;

    typedef pcl::PointIndices::Ptr PointIndicesPtr;
    typedef pcl::PointIndices::ConstPtr PointIndicesConstPtr;

    typedef typename pcl::search::Search<PointT> KdTree;
    typedef typename pcl::search::Search<PointT>::Ptr KdTreePtr;

    typedef pcl::VFHSignature308 VFHType;
    typedef pcl::PointCloud<VFHType> VFHCloudType;

    //typedef Matrix<float, 4, 1> Vector4f;

    classification (PointCloudPtr, std::vector<pcl::IndicesPtr>);

    /*
    * Save the classification results inside files in directory "histograms/"
    */
    void save();

    std::vector<float> getCardinality() {
        return cardinality_;
    }

    std::vector<float> getIntensity() {
        return intensity_;
    }

    std::vector<float> getNormals() {
        return norm_std_dev_;
    }

    std::vector<float> getCurvature() {
        return curv_std_dev_;
    }

    std::vector<float> getEigModule() {
        return eigModule_;
    }

    std::vector<VFHCloudType::Ptr> getVFH() {
        return vfh_ptrs_;
    }
    
    std::vector<pcl::SVMData> features;

    std::vector<double> cardinality_;
    std::vector<double> intensity_;
    std::vector<double> norm_std_dev_;
    std::vector<double> curv_std_dev_;
    std::vector<double> eigModule_;
    std::vector<double> density_;
    std::vector<double> firstEig_;
    std::vector<double> secondEig_;
    
    std::vector<VFHCloudType::Ptr> vfh_ptrs_;
    
private:
    /*
     * Extract clusters cardinalities
     * */
    void cardinality() {
        cardinality_.clear();
        for (int i=0; i< clusters_.size(); i++)
            // Normalized cardinality
            cardinality_.push_back( (double)clusters_[i]->size() / MAX_NORNALIZE);
    }

    /*
    * Extract clusters intensities
    * */
    void intensity() {
        intensity_.clear();
        for (int i=0; i< clusters_.size(); i++)
        {
            float buff=0;
            for (int j=0; j < clusters_[i]->size(); j++)
            {
                buff = buff + cloud_->points[ clusters_[i]->operator[](j) ].intensity;
            }
            // Normalized Intensity
            intensity_.push_back( (double)(buff / clusters_[i]->size()) );
        }
    }

    /*
    * Extract clusters normals and curvatures variance
    * */
    void normal_curv() {
        norm_std_dev_.clear();
        curv_std_dev_.clear();
        for (int i=0; i< clusters_.size(); i++)
        {
            float meanN[3]={0,0,0};
            std::vector<float> angles, curv;
            double stddevN=0, meanValN=0, stddevC=0, meanValC=0;

            // Calculate the mean normal value
            for (int j=0; j < clusters_[i]->size(); j++)
            {
                meanN[0] = meanN[0] + cluster_normals_[i]->points[j].normal[0];
                meanN[1] = meanN[1] + cluster_normals_[i]->points[j].normal[1];
                meanN[2] = meanN[2] + cluster_normals_[i]->points[j].normal[2];
            }
            meanN[0] = meanN[0] / clusters_[i]->size();
            meanN[1] = meanN[1] / clusters_[i]->size();
            meanN[2] = meanN[2] / clusters_[i]->size();

            // Angle difference from every point to the mean
            for (int j=0; j < clusters_[i]->size(); j++)
            {
                angles.push_back( cluster_normals_[i]->points[j].normal[0] * meanN[0] +
                                  cluster_normals_[i]->points[j].normal[1] * meanN[1] +
                                  cluster_normals_[i]->points[j].normal[2] * meanN[2] );
                curv.push_back( cluster_normals_[i]->points[j].curvature );
            }
            pcl::getMeanStd(angles, meanValN, stddevN);
            pcl::getMeanStd(curv, meanValC, stddevC);
            norm_std_dev_.push_back( stddevN );
            curv_std_dev_.push_back( stddevC );
        }
    }

    /*
    * Extract clusters EVD module
    * */
    void EVD() {
        eigModule_.clear();
        //float buff_EVD[clusters_.size()][3];
        float **eigenVal;
        eigenVal = new float*[clusters_.size()];
        //float eigenBuff[clusters_.size()][3];
        for (int i=0; i< clusters_.size(); i++)
        {
            eigenVal[i] = new float[3];
            Vector4f centroid;
            Matrix3f covMat;
            pcl::computeMeanAndCovarianceMatrix (*cloud_ , clusters_[i].operator*(), covMat, centroid);

            SelfAdjointEigenSolver<Matrix3f> eigensolver(covMat);
            eigenVal[i][0] = eigensolver.eigenvalues()[0];
            eigenVal[i][1] = eigensolver.eigenvalues()[1];
            eigenVal[i][2] = eigensolver.eigenvalues()[2];
            eigModule_.push_back( (double)sqrt( pow(eigenVal[i][0],2) + pow(eigenVal[i][1],2) + pow(eigenVal[i][2],2))  / MAX_NORNALIZE);
        }
    }

    /*
    * Calculate VFH
    * */
    void VFH() {
        vfh_ptrs_.clear();
        for (int i=0; i< clusters_.size(); ++i)
        {
	  VFHCloudType::Ptr vfh_temp (new VFHCloudType);
	  //std::cout << "Cluster " << i << " with " << clusters_[i]->size() << " points." << std::endl;
	  if(clusters_[i]->size() > 1000000){
	    vfh_temp.reset(new VFHCloudType);
	    //vfh_temp->points.clear();
	    vfh_ptrs_.push_back (vfh_temp);
	    continue;
	  }
           // VFHCloudType::Ptr vfh_temp (new VFHCloudType);
            vfher_.setIndices( clusters_[i] );
            vfher_.compute (*vfh_temp);
            vfh_ptrs_.push_back (vfh_temp);
        }
    }

    /*
    * Extract clusters density
    * */
    void density() {
        density_.clear();
        for (int i=0; i< clusters_.size(); i++)
        {
            Eigen::Vector4f minPt, maxPt;
            pcl::getMinMax3D(*cloud_, clusters_[i].operator*(), minPt, maxPt);

            float volume =0;
            volume = module( maxPt[0] - minPt[0] ) * module( maxPt[1] - minPt[1] ) * module( maxPt[2] - minPt[2]);
//             std::cout << maxPt[0] << " " << maxPt[1] << " " << maxPt[2] << " e ";
//             std::cout << minPt[0] << " " << minPt[1] << " " << minPt[2] << " " << std::endl;
            density_.push_back( (double)(volume / clusters_[i]->size()) );
        }
    }
    
    /*
    * Extract clusters PCA
    * */
    void pca() {
        firstEig_.clear();
	secondEig_.clear();
	
        pcl::PCA<pcl::PointXYZI> pca;

        for (int i=0; i< clusters_.size(); i++)
        {
            Eigen::Vector3f eigenVal(0.0, 0.0, 0.0);

            pca.setInputCloud(cloud_);
            pca.setIndices(clusters_[i]);
            if (clusters_[i]->size() > 2)
                eigenVal = pca.getEigenValues();

            firstEig_.push_back(eigenVal[0]/(eigenVal[0]+eigenVal[1]+eigenVal[2]));
            secondEig_.push_back(eigenVal[1]/(eigenVal[0]+eigenVal[1]+eigenVal[2]));

        }
    }
    std::vector<pcl::IndicesPtr> clusters_;
    PointCloudPtr cloud_;
    std::vector< pcl::PointCloud<pcl::Normal>::Ptr > cluster_normals_;
    pcl::VFHEstimation<PointT, pcl::Normal, VFHType> vfher_;
};

template <class PointT>
classification<PointT>::classification(PointCloudPtr cloud, std::vector<pcl::IndicesPtr> clusters) {
    cloud_=cloud;
    clusters_ = clusters;

    // Normals and curvatures estimation
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    typename pcl::search::KdTree<PointT>::Ptr tree (new typename pcl::search::KdTree<PointT> (false));
    ne.setInputCloud (cloud_);
    ne.setSearchMethod (tree);

    // Set up the VFHer class
     pcl::PointCloud<pcl::Normal>::Ptr normals_all (new pcl::PointCloud<pcl::Normal>);
     ne.setKSearch(50);
     ne.compute (*normals_all);
//     vfher_.setInputCloud (cloud_);
//     vfher_.setInputNormals (normals_all);
//     vfher_.setSearchMethod (tree);

    for (int i=0; i< clusters_.size(); i++)
    {
        // Normals
        pcl::PointCloud<pcl::Normal>::Ptr buffN (new pcl::PointCloud<pcl::Normal>);
	buffN->resize( clusters_[i]->size() );
	
	for(int j=0; j<clusters_[i]->size() ;j++){
	  buffN->points[j].normal_x = normals_all->points[ clusters_[i]->operator[](j) ].normal_x;
	  buffN->points[j].normal_y = normals_all->points[ clusters_[i]->operator[](j) ].normal_y;
	  buffN->points[j].normal_z = normals_all->points[ clusters_[i]->operator[](j) ].normal_z;
	}
	
        cluster_normals_.push_back(buffN);
	
	
// 	        // Normals
//         pcl::PointCloud<pcl::Normal>::Ptr buffN (new pcl::PointCloud<pcl::Normal>);
//         ne.setIndices( clusters[i] );
//         ne.setKSearch(50);
//         ne.compute (*buffN);
//         cluster_normals_.push_back(buffN);
    }

    features.clear();
    cardinality();
    intensity();
    normal_curv();
    EVD();
    density();
    pca();
    //VFH();
    
    //pcl::SVMData buff;
    
    for (int i=0; i< clusters_.size(); i++) {
        pcl::SVMData buff;
        buff.SV.resize(6);

       buff.SV[0].idx = 0;
       buff.SV[0].value = cardinality_[i];

        buff.SV[1].idx = 1;
        buff.SV[1].value = intensity_[i];

//        buff.SV[6].idx = 6;
//        buff.SV[6].value = norm_std_dev_[i];
//
//       buff.SV[3].idx = 3;
//       buff.SV[3].value = curv_std_dev_[i];

        buff.SV[2].idx = 2;
        buff.SV[2].value = eigModule_[i];

        buff.SV[3].idx = 3;
        buff.SV[3].value = density_[i];

        buff.SV[4].idx = 4;
        buff.SV[4].value = firstEig_[i];

        buff.SV[5].idx = 5;
        buff.SV[5].value = secondEig_[i];

        features.push_back(buff);
    }
 //std::cout << features.size() << std::endl;

//     std::cout << "\nMaximum cardinality_: " << *(std::max_element(cardinality_.begin(),cardinality_.end()) ) << std::endl;
//     std::cout << "Maximum intensity_: " << *(std::max_element(intensity_.begin(),intensity_.end()) ) << std::endl;
//     std::cout << "Maximum norm_std_dev_: " << *(std::max_element(norm_std_dev_.begin(),norm_std_dev_.end()) ) << std::endl;
//     std::cout << "Maximum curv_std_dev_: " << *(std::max_element(curv_std_dev_.begin(),curv_std_dev_.end()) ) << std::endl;
//     std::cout << "Maximum eigModule: " << *(std::max_element(eigModule_.begin(),eigModule_.end()) ) << std::endl;

}

template <class PointT>
void classification<PointT>::save()
{
    system("mkdir histograms");
    FILE *fc, *fi, *fn, *fcu, *fev;
    fc = fopen ( "histograms/cardinality.txt", "wt" ) ;
    fi = fopen ( "histograms/intensity.txt", "wt" ) ;
    fn = fopen ( "histograms/normals.txt", "wt" ) ;
    fcu = fopen ( "histograms/curvature.txt", "wt" ) ;
    fev = fopen ( "histograms/eigenvalue.txt", "wt" ) ;
    for (int i=0; i< clusters_.size(); i++) {
        fprintf ( fc, " %f\n", cardinality_[i] );
        fprintf ( fi, " %f\n", intensity_[i] );
        fprintf ( fn, " %f\n", norm_std_dev_[i] );
        fprintf ( fcu, "%f\n", curv_std_dev_[i] );
        fprintf ( fev, "%f\n", eigModule_[i] );
    }
    fclose(fc);
    fclose(fi);
    fclose(fn);
    fclose(fcu);
    fclose(fev);
}
