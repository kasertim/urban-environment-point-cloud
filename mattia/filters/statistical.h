#include "pcl/point_types.h"
#include "pcl/filters/filter.h"
#include "pcl/search/pcl_search.h"
#include "pcl/common/common.h"
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

namespace pcl
{
/** \brief @b StatisticalOutlierRemoval uses point neighborhood statistics to filter outlier data. For more
  * information check:
  *   - R. B. Rusu, Z. C. Marton, N. Blodow, M. Dolha, and M. Beetz.
  *     Towards 3D Point Cloud Based Object Maps for Household Environments
  *     Robotics and Autonomous Systems Journal (Special Issue on Semantic Knowledge), 2008.
  *
  * \note setFilterFieldName (), setFilterLimits (), and setFilterLimitNegative () are ignored.
  * \author Radu Bogdan Rusu
  * \ingroup filters
  */
template<typename PointT>
class StatisticalOutlierRemoval : public Filter<PointT>
{
    using Filter<PointT>::input_;
    using Filter<PointT>::indices_;
    using Filter<PointT>::filter_name_;
    using Filter<PointT>::getClassName;

    using Filter<PointT>::removed_indices_;
    using Filter<PointT>::extract_removed_indices_;

    typedef typename pcl::search::Search<PointT> KdTree;
    typedef typename pcl::search::Search<PointT>::Ptr KdTreePtr;

    typedef typename Filter<PointT>::PointCloud PointCloud;
    typedef typename PointCloud::Ptr PointCloudPtr;
    typedef typename PointCloud::ConstPtr PointCloudConstPtr;

    typedef Matrix<float, 3, 1> Vector3f;

public:
    /** \brief Empty constructor. */
    StatisticalOutlierRemoval (bool extract_removed_indices = false) :
            Filter<PointT>::Filter (extract_removed_indices), mean_k_ (2), std_mul_ (0.0), tree_ (), negative_ (false)
    {
        filter_name_ = "StatisticalOutlierRemoval";
    }

    /** \brief Set the number of points (k) to use for mean distance estimation
      * \param nr_k the number of points to use for mean distance estimation
      */
    inline void
    setMeanK (int nr_k)
    {
        mean_k_ = nr_k;
    }

    /** \brief Get the number of points to use for mean distance estimation. */
    inline int
    getMeanK ()
    {
        return (mean_k_);
    }

    /** \brief Set the standard deviation multiplier threshold. All points outside the
      * \f[ \mu \pm \sigma \cdot std\_mul \f]
      * will be considered outliers, where \f$ \mu \f$ is the estimated mean,
      * and \f$ \sigma \f$ is the standard deviation.
      * \param std_mul the standard deviation multiplier threshold
      */
    inline void
    setStddevMulThresh (double std_mul)
    {
        std_mul_ = std_mul;
    }

    /** \brief Get the standard deviation multiplier threshold as set by the user. */
    inline double
    getStddevMulThresh ()
    {
        return (std_mul_);
    }

    /** \brief Set whether the inliers should be returned (true), or the outliers (false).
      * \param negative true if the inliers should be returned, false otherwise
      */
    inline void
    setNegative (bool negative)
    {
        negative_ = negative;
    }

    /** \brief Get the value of the internal negative_ parameter. If
      * true, all points \e except the input indices will be returned.
      */
    inline bool
    getNegative ()
    {
        return (negative_);
    }
    void
    applyFilterMat (PointCloud &output);
    void
    applyFilterMat2 (PointCloud &output);

protected:
    /** \brief The number of points to use for mean distance estimation. */
    int mean_k_;

    /** \brief Standard deviations threshold (i.e., points outside of
     * \f$ \mu \pm \sigma \cdot std\_mul \f$ will be marked as outliers). */
    double std_mul_;

    /** \brief A pointer to the spatial search object. */
    KdTreePtr tree_;

    /** \brief If true, the outliers will be returned instead of the inliers (default: false). */
    bool negative_;

    /** \brief Apply the filter
      * \param output the resultant point cloud message
      */
    void
    applyFilter (PointCloud &output);

    void
    covarianceMatrix (PointT point, std::vector<int> nn_indices, Matrix3f *cov_mat);

};
}

template <typename PointT> void
pcl::StatisticalOutlierRemoval<PointT>::applyFilter (PointCloud &output)
{
    if (std_mul_ == 0.0)
    {
        PCL_ERROR ("[pcl::%s::applyFilter] Standard deviation multiplier not set!\n", getClassName ().c_str ());
        output.width = output.height = 0;
        output.points.clear ();
        return;
    }

    if (input_->points.empty ())
    {
        output.width = output.height = 0;
        output.points.clear ();
        return;
    }

    // Initialize the spatial locator
    if (!tree_)
    {
        if (input_->isOrganized ())
            tree_.reset (new pcl::search::OrganizedNeighbor<PointT> ());
        else
            tree_.reset (new pcl::search::KdTree<PointT> (false));
    }

    // Send the input dataset to the spatial locator
    tree_->setInputCloud (input_);

    // Allocate enough space to hold the results
    std::vector<int> nn_indices (indices_->size ());
    std::vector<float> nn_dists (indices_->size ());

    std::vector<float> distances (indices_->size ());
    
    output.points.resize (input_->points.size ());      // reserve enough space
    int nr_p = 0;
    
    // Go over all the points and calculate the mean or smallest distance
    for (size_t cp = 0; cp < indices_->size (); ++cp)
    {
      nn_indices.clear();
        if (!pcl_isfinite (input_->points[(*indices_)[cp]].x) ||
                !pcl_isfinite (input_->points[(*indices_)[cp]].y) ||
                !pcl_isfinite (input_->points[(*indices_)[cp]].z))
        {
            distances[cp] = 0;
            continue;
        }

        if (//tree_->nearestKSearch ((*indices_)[cp], mean_k_, nn_indices, nn_dists) == 0
	  tree_->radiusSearch ((*indices_)[cp], mean_k_, nn_indices, nn_dists) == 0
	)
        {
            distances[cp] = 0;
            PCL_WARN ("[pcl::%s::applyFilter] Searching for the closest %d neighbors failed.\n", getClassName ().c_str (), mean_k_);
            continue;
        }
        Matrix3f cov_mat;
        cov_mat << 0,0,0,
        0,0,0,
        0,0,0;

        covarianceMatrix(input_->points[(*indices_)[cp]], nn_indices, &cov_mat);

        SelfAdjointEigenSolver<Matrix3f> eigensolver(cov_mat);

        //distances[cp] =eigensolver.eigenvalues().minCoeff();
	if(std_mul_ > (float)(eigensolver.eigenvalues().minCoeff() / eigensolver.eigenvalues().maxCoeff()))
	  output.points[nr_p++] = input_->points[(*indices_)[cp]];
	

 //	if(cp > 0 && distances[cp] != distances[cp] )//distances[cp]!= distances[cp-1])
 //	   std::cout << "min eig " << distances[cp] << std::endl << std::endl << cov_mat << endl << endl;
    }


    //removed_indices_->resize (input_->points.size ());

    // Build a new cloud by neglecting outliers
//     int nr_p = 0;
//     int nr_removed_p = 0;
// 
// //     for (size_t cp = 0; cp < indices_->size (); ++cp)
// //     {
// //         output.points[nr_p++] = input_->points[(*indices_)[cp]];
// //     }

    output.points.resize (nr_p);
    output.width  = nr_p;
    output.height = 1;
    output.is_dense = true; // nearestKSearch filters invalid points

//     removed_indices_->resize (nr_removed_p);
}


/**
 * \brief The function calculates the covariance matrix
 * \param point the query point
 * \param nn_indices indices of the neighbour points
 * \param cov_mat the output covariance matrix
 **/
template <typename PointT> void
pcl::StatisticalOutlierRemoval<PointT>::covarianceMatrix (PointT point, std::vector<int> nn_indices, Matrix3f *cov_mat)
{

      Vector3f mu(0,0,0);
//        point.x,
//        point.y,
//        point.z
//     );
      
  // Determine the centroid
   for (int j = 0; j < nn_indices.size(); j++)
    {
        Vector3f x_buff(
            input_->points[(*indices_)[nn_indices[j]] ].x ,
            input_->points[(*indices_)[nn_indices[j]] ].y ,
            input_->points[(*indices_)[nn_indices[j]] ].z 
        );
	mu = mu + x_buff;
    }
    mu = mu / nn_indices.size();
   int count=0;
    
    for (int j = 0; j < nn_indices.size(); j++)
    {
        Vector3f x_i(
            input_->points[(*indices_)[nn_indices[j]] ].x ,
            input_->points[(*indices_)[nn_indices[j]] ].y ,
            input_->points[(*indices_)[nn_indices[j]] ].z 
        );

        if (x_i == mu) 
	  continue;
	count++;

        (*cov_mat) += (( x_i - mu ) * ( x_i - mu ).transpose()) ;
    }
    (*cov_mat) =  (*cov_mat) / count;
}



