#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <pcl/filters/radius_outlier_removal.h>

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
    pcl::io::loadPCDFile ("benchmark_XYZIC_no_wall.pcd", *cloud_in);
    std::cout << "Input: " << cloud_in->points.size () << std::endl;

    boost::shared_ptr<const std::vector<int> > indices;

    pcl::RadiusOutlierRemoval<PointType> outrem (true);
    outrem.setInputCloud(cloud_in);
    outrem.setRadiusSearch(300.0);
    outrem.setMinNeighborsInRadius (150);
    outrem.filter (*cloud_out);
    indices = outrem.getRemovedIndices();

    // Visualize which ones are actually being removed
//    *cloud_out = *cloud_in;
//    for (int i = 0; i < indices->size (); ++i)
//      cloud_out->points[(*indices)[i]].intensity = 10000.0;

    std::cout << "Output: " << cloud_out->points.size () << std::endl;
    pcl::io::savePCDFileBinary ("benchmark_XYZIC_no_isolated.pcd", *cloud_out);
  }
};

