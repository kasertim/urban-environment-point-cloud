// Boost includes. Needed everywhere.
#include <boost/shared_ptr.hpp>

// Point Cloud message includes. Needed everywhere.
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/PointIndices.h>
#include <pcl/console/print.h>
#include <pcl/pcl_base.h>
#include "pcl/search/pcl_search.h"
#include <pcl/io/pcd_io.h>


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
class RegionGrowing : public PCLBase<PointT>
{
public:
    typedef pcl::PointCloud<PointT> PointCloud;
    typedef typename PointCloud::Ptr PointCloudPtr;
    typedef typename PointCloud::ConstPtr PointCloudConstPtr;

    typedef PointIndices::Ptr PointIndicesPtr;
    typedef PointIndices::ConstPtr PointIndicesConstPtr;

    typedef typename pcl::search::Search<PointT> KdTree;
    typedef typename pcl::search::Search<PointT>::Ptr KdTreePtr;



    /** \brief Provide a pointer to the input dataset
      * \param cloud the const boost shared pointer to a PointCloud message
      */
    virtual inline void
    setInputCloud ( const PointCloudConstPtr &cloud )
    {
        input_ = cloud;
	eps_angle_ = 0;
    }

    /** \brief Get a pointer to the input point cloud dataset. */
    inline PointCloudConstPtr const
    getInputCloud ()
    {
        return ( input_ );
    }

    virtual inline void
    setGrowingDistance ( float dist )
    {
        dist_ = dist;
    }

    virtual inline void
    setNormals (const pcl::PointCloud<pcl::Normal>::Ptr normals)
    {
      normals_ = normals;
      //pcl::copyPointCloud<pcl::Normal, pcl::Normal>(normals, normals_); 
    }
    
    virtual inline void
    setEpsAngle (double eps_angle)
    {
      eps_angle_ = eps_angle;
    }
//     virtual inline std::vector<PointIndices>
//     getOutput (std::vector<PointIndices> output)
//     {
//         //output = output_;
// 	output.operator=(output_);
//     }

    virtual inline void cluster (std::vector<pcl::IndicesPtr> *output);


protected:
    /** \brief A pointer to the spatial search object. */
    KdTreePtr tree_;

    PointCloudConstPtr input_;
    float dist_;
    pcl::PointCloud<pcl::Normal>::Ptr normals_;
    double eps_angle_;
    //std::vector<PointIndices> output_;
};
}

template <typename PointT>
void pcl::RegionGrowing<PointT>::cluster (std::vector<pcl::IndicesPtr> *output) {

  output->clear();
  std::vector<bool> processed ( input_->points.size (), false );
  std::vector<int> nn_indices;
  std::vector<float> nn_distances;

    // Initialize the spatial locator
    KdTreePtr tree;
        if (input_->isOrganized ())
            tree.reset (new pcl::search::OrganizedNeighbor<PointT> ());
        else
            tree.reset (new pcl::search::KdTree<PointT> (false));
    
        // Send the input dataset to the spatial locator
    tree->setInputCloud (input_);

    for ( size_t i = 0; i < input_->points.size (); ++i )
    {
        if ( processed[i] ) continue;

        pcl::IndicesPtr seed_queue (new std::vector<int>);
        int sq_idx = 0;
        seed_queue->push_back ( i );

        processed[i] = true;

        while ( sq_idx < ( int ) seed_queue->size () )
        {
            // Search for sq_idx
            nn_indices.clear();
            if ( !tree->radiusSearch ( seed_queue->operator[](sq_idx), dist_, nn_indices, nn_distances ) )
            {
                sq_idx++;
                continue;
            }

            for ( size_t j = 0; j < nn_indices.size (); ++j )
            {
                if ( processed[nn_indices[j]] )                       // Has this point been processed before ?
                    continue;
                if ( eps_angle_ ) {
                    double dot_p = normals_->points[i].normal[0] * normals_->points[nn_indices[j]].normal[0] +
                                   normals_->points[i].normal[1] * normals_->points[nn_indices[j]].normal[1] +
                                   normals_->points[i].normal[2] * normals_->points[nn_indices[j]].normal[2];
                    if ( fabs (acos (dot_p)) < eps_angle_ )
                    {
                        processed[nn_indices[j]] = true;
                        seed_queue->push_back ( nn_indices[j] );
                    }
                    continue;
                }
                else {
                    processed[nn_indices[j]] = true;
                    seed_queue->push_back ( nn_indices[j] );
                }

            }

            sq_idx++;
        }
        // If this queue is satisfactory, add to the clusters
        if ( true )
        {
	  output->push_back(seed_queue);
//             pcl::PointIndices r;
//             r.indices.resize ( seed_queue.size () );
//             for ( size_t j = 0; j < seed_queue.size (); ++j )
//             {
//                 r.indices[j] = seed_queue[j];
//             }
        }
    }
    //std::cout << "clusters? " <<output_.size()<< std::endl;


}
template <class PointT>
void saveclusters(std::vector<pcl::IndicesPtr> indices, PointT cloud, std::string path){
  for (int i=0; i<indices.size(); i++)
  {
    std::stringstream name;
    name << path << i << ".pcd";
    pcl::PointCloud<pcl::PointXYZI> ciao;
    pcl::io::savePCDFile<pcl::PointXYZI>(name.str().data(), *cloud,  indices[i].operator*(),1);
    //, (std::vector<int>*)&indices[i],1
  }
}