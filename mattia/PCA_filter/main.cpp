#include <iostream>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include "edge_contour_extraction.hpp"

using namespace std;
typedef pcl::PointXYZ PointType;

int main(int argc, char **argv) {
  
  if(argc < 4)
  {
    std::cout << "Please specify input cloud, number of neighbours and tolerance" << std::endl;
    std::cout << "  ex: "<< argv[0] << " table.pcd 100 0.66" << std::endl;
  }
  
  pcl::PointCloud<PointType>::Ptr cloud_ (new pcl::PointCloud<PointType>);
  
      // Load the clouds
    if (pcl::io::loadPCDFile (argv[1], *cloud_))
        return 0;

    pcl::EdgeContourExtraction<PointType> pr;
    pr.setInputCloud(cloud_);
    pr.setEigMeanK(atof(argv[2]));
    pr.setPrincEigThresh(atof(argv[3]));
    //pr.setNegative(1);
    pr.filter(*cloud_);
    
    std::string strOut, command;
    strOut.assign("out_");
    strOut.append(argv[1]);
    pcl::io::savePCDFileBinary(strOut.data(),*cloud_);
    
    command.assign("pcd_viewer -multiview 1 ");
    command.append(argv[1]);
    command.append(" ");
    command.append(strOut.data());
    system(command.data());
    
    return 0;
}
