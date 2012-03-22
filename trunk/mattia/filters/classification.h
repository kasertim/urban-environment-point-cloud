#include <boost/graph/graph_concepts.hpp>
#include <pcl/point_cloud.h>
#include <pcl/PointIndices.h>
#include <pcl/console/print.h>
#include <pcl/pcl_base.h>
#include "pcl/search/pcl_search.h"
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>

#include <Eigen/Dense>

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
    
    //typedef Matrix<float, 4, 1> Vector4f;
    //template <class PointType> 
    classification (PointCloudPtr, std::vector<pcl::IndicesPtr>);

private:
    void cardinality() {
        for (int i=0; i< clusters_.size(); i++)
            cardinality_.push_back( clusters_[i]->size() );
    }

    void intensity() {
        for (int i=0; i< clusters_.size(); i++)
        {
            float buff=0;
            for (int j=0; j < clusters_[i]->size(); j++)
            {
                buff = buff + cloud_->points[ clusters_[i]->operator[](j) ].intensity;
            }
            intensity_.push_back( buff / clusters_[i]->size() );
        }
    }
    
    void normal_curv() {
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
    
    void EVD() {
      //float buff_EVD[clusters_.size()][3];
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
        }
        //eigenVal = eigenBuff;
        //eigenVal = buff_EVD;
    }

    std::vector<pcl::IndicesPtr> clusters_;
    PointCloudPtr cloud_;
    //pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_;
    std::vector< pcl::PointCloud<pcl::Normal>::Ptr > cluster_normals_;
//     std::vector< pcl::PointCloud<pcl::PointXYZINormal>::Ptr > cluster_curvatures_;

    std::vector<float> cardinality_;
    std::vector<float> intensity_;
    std::vector<float> norm_std_dev_;
    std::vector<float> curv_std_dev_;
    float **eigenVal;
};

template <class PointT>
classification<PointT>::classification(PointCloudPtr cloud, std::vector<pcl::IndicesPtr> clusters) {
    cloud_=cloud;
    clusters_ = clusters;

    // Normals estimation
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI> ());
    ne.setInputCloud (cloud_);
    ne.setSearchMethod (tree);
    
    // Curvature estimation
//     pcl::PrincipalCurvaturesEstimation<pcl::PointXYZI, pcl::Normal, pcl::PointXYZINormal> pce;
//     pce.setInputCloud(cloud_);
//     pce.setSearchMethod (tree);
    
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
        fprintf ( fev, "%f %f %f\n", eigenVal[i][0], eigenVal[i][1], eigenVal[i][2] );
    }
    fclose(fc);
    fclose(fi);
    fclose(fn);
    fclose(fcu);
    fclose(fev);
}