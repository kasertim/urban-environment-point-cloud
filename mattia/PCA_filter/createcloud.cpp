#include <iostream>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
//#include "pca_removal.cpp"

using namespace std;
typedef pcl::PointXYZI PointType;

int main(int argc, char **argv) {

    pcl::PointCloud<PointType>::Ptr cloud_ (new pcl::PointCloud<PointType>);

    PointType buff;
    buff.x=-800;
    buff.z = -2000;
    buff.intensity = 0;
    // create line
    for (int i=0; i < 2000; i++) {
        buff.y = i/4 ;
        cloud_->push_back(buff);
    }

    buff.intensity = 65;
    // create thik line
    for (int i=0; i < 2000; i++) {
        buff.y = i/4 ;
        buff.x=-350;
        for (int j=0; j<20; j++) {
            buff.x=-500+j;
            cloud_->push_back(buff);
        }
    }

    buff.intensity = 130;
    //create plane
    for (int i=0; i < 500; i=i+2) {
        buff.y = i;
        for (int j=0; j < 500; j=j+2) {
            buff.x = j;
            cloud_->push_back(buff);
        }
    }

    //create solid
    buff.intensity = 200;
    for (int i=0; i < 500; i=i+8) {
        buff.y = i;
        for (int j=0; j < 500; j=j+8) {
            buff.x = j+1000;
            for (int k=0; k < 500; k=k+8) {
                buff.z = k-2000;
                cloud_->push_back(buff);
            }
        }
    }

    pcl::io::savePCDFileBinary("figures_out.pcd", *cloud_);

    return 0;
}

