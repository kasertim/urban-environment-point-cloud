#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> Cloud;
typedef Cloud::Ptr CloudPtr;

class Vluishanden
{
public:
  Vluishanden ()
  {
    CloudPtr cloud_in (new Cloud);
    CloudPtr cloud_out (new Cloud);
    pcl::io::loadPCDFile ("benchmark_XYZIC_no_ground.pcd", *cloud_in);
    std::cout << "Input: " << cloud_in->points.size () << std::endl;

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

    // Create the segmentation object
    pcl::SACSegmentation<PointType> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (200.0);
    seg.setInputCloud (cloud_in);
    seg.segment (*inliers, *coefficients);

    if (inliers->indices.size () == 0)
    {
      PCL_ERROR("Could not estimate a planar model for the given dataset.");
      return;
    }
    std::cout << "Model coefficients: " << coefficients->values[0] << " " << coefficients->values[1] << " " << coefficients->values[2] << " " << coefficients->values[3] << std::endl;

    pcl::ExtractIndices<PointType> filter;
    filter.setInputCloud (cloud_in);
    filter.setIndices (inliers);
    filter.setNegative (true);
    filter.filter (*cloud_out);

    std::cout << "Output: " << cloud_out->points.size () << std::endl;
    pcl::io::savePCDFileBinary ("benchmark_XYZIC_no_wall.pcd", *cloud_out);
  }
};

