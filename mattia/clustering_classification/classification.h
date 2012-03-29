#include <boost/graph/graph_concepts.hpp>
#include <pcl/point_cloud.h>
#include <pcl/PointIndices.h>
#include <pcl/console/print.h>
#include <pcl/pcl_base.h>
#include "pcl/search/pcl_search.h"
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>

// I added these
#include <pcl/common/common.h>
#include <pcl/features/vfh.h>

#include <Eigen/Dense>
using namespace Eigen;

#define MAX_NORNALIZE 1//50000000.0

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

    std::vector<double> cardinality_;
    std::vector<double> intensity_;
    std::vector<double> norm_std_dev_;
    std::vector<double> curv_std_dev_;
    std::vector<double> eigModule_;
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
            vfher_.setIndices( clusters_[i] );
            vfher_.compute (*vfh_temp);
            vfh_ptrs_.push_back (vfh_temp);
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
    typename pcl::search::KdTree<PointT>::Ptr tree (new typename pcl::search::KdTree<PointT> ());
    ne.setInputCloud (cloud_);
    ne.setSearchMethod (tree);

    // Set up the VFHer class
    pcl::PointCloud<pcl::Normal>::Ptr normals_all (new pcl::PointCloud<pcl::Normal>);
    ne.setKSearch(20);
    ne.compute (*normals_all);
    vfher_.setInputCloud (cloud_);
    vfher_.setInputNormals (normals_all);
    vfher_.setSearchMethod (tree);

    for (int i=0; i< clusters_.size(); i++)
    {
        // Normals
        pcl::PointCloud<pcl::Normal>::Ptr buffN (new pcl::PointCloud<pcl::Normal>);
        ne.setIndices( clusters[i] );
        ne.setKSearch(20);
        ne.compute (*buffN);
        cluster_normals_.push_back(buffN);
    }

    cardinality();
    intensity();
    normal_curv();
    EVD();
    VFH();


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
