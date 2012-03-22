#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <pcl/features/normal_3d.h>

struct EIGEN_ALIGN16 PointXYZINormalWeighting
{
  PCL_ADD_POINT4D;
  PCL_ADD_NORMAL4D;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  union
  {
    struct
    {
      float intensity;
      float curvature;
      float weighting;
    };
    float data_c[4];
  };
};

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZINormalWeighting,
    (float, x, x) (float, y, y) (float, z, z) (float, normal_x, normal_x) (float, normal_y, normal_y) (float, normal_z, normal_z) (float, intensity, intensity) (float, curvature, curvature) (float, weighting, weighting))

typedef pcl::PointXYZI InputPointType;
typedef PointXYZINormalWeighting OutputPointType;
typedef pcl::Normal NormalType;

typedef pcl::PointCloud<InputPointType> InputCloud;
typedef pcl::PointCloud<OutputPointType> OutputCloud;
typedef pcl::PointCloud<NormalType> NormalCloud;
typedef pcl::search::KdTree<InputPointType> SearcherType;

int
main (int argc, char** argv)
{
    // Read input PCD
    InputCloud::Ptr cloud_in (new InputCloud);
    OutputCloud::Ptr cloud_out (new OutputCloud);
    NormalCloud::Ptr normals (new NormalCloud);
    SearcherType::Ptr searcher (new SearcherType);
    pcl::io::loadPCDFile ("input.pcd", *cloud_in);
    std::cout << "Input: " << cloud_in->points.size () << std::endl;

    // Intensity normalization (make it always from 0.0 to 1.0)
    float intensity_min = std::numeric_limits<float>::infinity (), intensity_max = -std::numeric_limits<float>::infinity ();
    for (int p_it = 0; p_it < static_cast<int> (cloud_in->points.size ()); ++p_it)
    {
      if (cloud_in->points[p_it].intensity < intensity_min)
        intensity_min = cloud_in->points[p_it].intensity;
      if (cloud_in->points[p_it].intensity > intensity_max)
        intensity_max = cloud_in->points[p_it].intensity;
    }
    float scale = 1.0 / (intensity_max - intensity_min);
    for (int p_it = 0; p_it < static_cast<int> (cloud_in->points.size ()); ++p_it)
      cloud_in->points[p_it].intensity = (cloud_in->points[p_it].intensity - intensity_min) * scale;

    // Normal estimation
    pcl::NormalEstimation<InputPointType, NormalType> ne;
    ne.setInputCloud (cloud_in);
    ne.setSearchMethod (searcher);
    ne.setKSearch (20);
    ne.compute (*normals);

    // Concatenate the different information into the output
    copyPointCloud (*cloud_in, *cloud_out);
    copyPointCloud (*normals, *cloud_out);

    // Write output PCD
    std::cout << "Output: " << cloud_out->points.size () << std::endl;
    pcl::io::savePCDFileBinary ("output.pcd", *cloud_out);
    return (0);
}

