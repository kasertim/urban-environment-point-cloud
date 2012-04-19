#include <iostream>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include "edge_contour_extraction.hpp"

using namespace std;
typedef pcl::PointXYZ PointType;

int main(int argc, char **argv) {
    
  pcl::PointCloud<PointType>::Ptr cloud_ (new pcl::PointCloud<PointType>);
  
      // Load the clouds
    if (pcl::io::loadPCDFile (argv[1], *cloud_))
        return 0;
    
//     pcl::RadiusOutlierRemoval<PointType> ror;
//     ror.setInputCloud(cloud_);
//     ror.setRadiusSearch(0.01);
//     ror.setMinNeighborsInRadius(10);
//     ror.filter(*cloud_);
    /*
    pcl::io::savePCDFile("table_out.pcd",*cloud_);*/
   
    pcl::EdgeContourExtraction<PointType> pr;
    pr.setInputCloud(cloud_);
    pr.setEigMeanK(atof(argv[2]));
    pr.setPrincEigThresh(atof(argv[3]));
    //pr.setNegative(1);
    pr.filter(*cloud_);
    
    std::string ciaoZio, command;
    ciaoZio.assign("out_");
    ciaoZio.append(argv[1]);
    pcl::io::savePCDFileBinary(ciaoZio.data(),*cloud_);
    
    command.assign("pcd_viewer -multiview 1 ");
    command.append(argv[1]);
    command.append(" ");
    command.append(ciaoZio.data());
    system(command.data());
    
    return 0;
}
