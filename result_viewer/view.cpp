#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>

int
main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::IndicesPtr indices (new std::vector<int>);
  pcl::ExtractIndices<pcl::PointXYZI> filter;
  pcl::PointCloud<pcl::PointXYZI> cloud_int;
  pcl::PointCloud<pcl::PointXYZRGBA> cloud_out;

  pcl::io::loadPCDFile (argv[1], *cloud_in);
  indices->resize (cloud_in->points.size () / 25);
  for (size_t i = 0; i < indices->size (); ++i)
    (*indices)[i] = i * 25;
  filter.setInputCloud (cloud_in);
  filter.setIndices (indices);
  filter.filter (cloud_int);
  pcl::copyPointCloud (cloud_int, cloud_out);
  for (int i = 0; i < cloud_out.points.size (); ++i)
  {
    cloud_out.points[i].r = cloud_int.points[i].intensity;
    cloud_out.points[i].g = cloud_int.points[i].intensity;
    cloud_out.points[i].b = cloud_int.points[i].intensity;
  }
  pcl::io::savePCDFileBinary ("multi1.pcd", cloud_out);

  pcl::io::loadPCDFile (argv[2], *cloud_in);
  indices->resize (cloud_in->points.size () / 25);
  for (size_t i = 0; i < indices->size (); ++i)
    (*indices)[i] = i * 25;
  filter.setInputCloud (cloud_in);
  filter.setIndices (indices);
  filter.filter (cloud_int);
  pcl::copyPointCloud (cloud_int, cloud_out);
  for (int i = 0; i < cloud_out.points.size (); ++i)
  {
    cloud_out.points[i].r = cloud_int.points[i].intensity;
    cloud_out.points[i].g = cloud_int.points[i].intensity;
    cloud_out.points[i].b = cloud_int.points[i].intensity;
  }
  pcl::io::savePCDFileBinary ("multi2.pcd", cloud_out);

  system ("pcl_pcd_viewer -multiview 1 multi1.pcd multi2.pcd");
  return (0);
}
