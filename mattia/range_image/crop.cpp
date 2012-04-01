#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

using namespace std;
typedef pcl::PointXYZI PointType;

int main(int argc, char **argv) {

    pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud_out (new pcl::PointCloud<PointType>);
    float xmin, xmax, ymin, ymax;

    if (argc < 7)
    {
        cerr << "\nThis program extract a portion of cloud from an organized one: " << endl;
        cerr << "\n  usage: " << argv[0] << " input.pcd output.pcd x_min[%] x_max[%] y_min[%] y_max[%]" << endl;
        cerr << "  - x_min, x_max: minimum and maximum percentage of the cloud to be taken along X-axis" << endl;
        cerr << "  - y_min, y_max: minimum and maximum percentage of the cloud to be taken along Y-axis\n" << endl;
        cerr << "  ex:    "<< argv[0] << " cloud.pcd output.pcd 10 50 10 50\n" << endl;
        return 0;
    }

    if (pcl::io::loadPCDFile (argv[1], *cloud))
        return 0;

    xmin = atof(argv[3]);
    xmax = atof(argv[4]);
    ymin = atof(argv[5]);
    ymax = atof(argv[6]);

    if (!cloud->isOrganized()) {
        cerr << "Input cloud not organized. Quitting..." << endl;
        return 0;
    }
    if (xmin < 0 || xmin > 100 || xmin >= xmax || xmax < 0 || xmax > 100) {
        cerr << "X param not valid"<< endl;
        return 0;
    }
    if (ymin < 0 || ymin > 100 || ymin >= ymax || ymax < 0 || ymax > 100) {
        cerr << "Y param not valid"<< endl;
        return 0;
    }
    int a=cloud->width / 100 * xmin;
    int b=cloud->width / 100 * xmax;
    int c=cloud->height / 100 * ymin;
    int d=cloud->height / 100 * ymax;
    int width=cloud->width;
    //cout << cloud->at(a,b) << " and "<< cloud->points[(cloud->width) * b + a] << endl;

    cloud_out->width = b-a ;
    cloud_out->height = d-c;
    cloud_out->points.resize( cloud_out->width * cloud_out->height);
    
    for(int i=0; i < cloud_out->height; i++)
      for(int j=0; j < cloud_out->width; j++){
	cloud_out->points[cloud_out->width * i + j].x =  cloud->points[cloud->width * (i+c) + (j+a)].x;
	cloud_out->points[cloud_out->width * i + j].y =  cloud->points[cloud->width * (i+c) + (j+a)].y;
	cloud_out->points[cloud_out->width * i + j].z =  cloud->points[cloud->width * (i+c) + (j+a)].z;
	cloud_out->points[cloud_out->width * i + j].intensity =  cloud->points[cloud->width * (i+c) + (j+a)].intensity;
      }

    pcl::io::savePCDFileBinary(argv[2], *cloud_out);
    return 0;
}