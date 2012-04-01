#include <iostream>
#include <string>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <cv.h>
#include <highgui.h>

using namespace pcl;

int main(int argc, char **argv) {
  
  pcl::PointCloud<pcl::PointXYZI>::Ptr pointCloud (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::io::loadPCDFile(argv[1], *pointCloud);
  
  // We now want to create a range image from the above point cloud, with a 1deg angular resolution
//   float angularResolution = (float) (  1.0f * (M_PI/180.0f));  //   1.0 degree in radians
//   float maxAngleWidth     = (float) (360.0f * (M_PI/180.0f));  // 360.0 degree in radians
//   float maxAngleHeight    = (float) (180.0f * (M_PI/180.0f));  // 180.0 degree in radians
//   Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
//   pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
//   float noiseLevel=0.00;
//   float minRange = 0.0f;
//   int borderSize = 1;
//   
//   pcl::RangeImage rangeImage;
//   rangeImage.createFromPointCloud(*pointCloud, angularResolution, maxAngleWidth, maxAngleHeight,
//                                   sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);
//   
//   std::cout << rangeImage << "\n";
    
  IplImage *img = cvCreateImage( cvSize( pointCloud->width, pointCloud->height ), IPL_DEPTH_16S, 3 );
  CvScalar s;
  for(int j=0; j<pointCloud->width; j++)
    for(int i=0; i<pointCloud->height; i++){
      if(isFinite(pointCloud->at(j,i))){
	s.val[0]=pointCloud->at(j,i).intensity;
        s.val[1]=pointCloud->at(j,i).intensity;
        s.val[2]=pointCloud->at(j,i).intensity;
      } else {
	s.val[0]=0;
	s.val[1]=0;
	s.val[2]=255;
      }
      cvSet2D(img,i,j,s);
    }
    
  std::string output;
  output.assign(argv[1]);
  output.append(".png");
  cvSaveImage(output.data(),img);
   
    return 0;
}
