#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "pcl/octree/octree.h"

typedef pcl::PointXYZI PointType;
typedef pcl::PointCloud<PointType> Cloud;
typedef Cloud::Ptr CloudPtr;

bool
checkPointInRange (PointType point, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
{
  if (point.x > xmin && point.x < xmax && point.y > ymin && point.y < ymax && point.z > zmin && point.z < zmax)
    return true;
  else
    return false;
}

int
main (int argc, char** argv)
{
  if (argc < 2)
  {
    std::cerr << "Input argument 1: a .pcd file to compare against benchmark-target.pcd" << std::endl;
    std::cerr << "Input argument 2 (optional): the octree resolution to use (default = 1.0)" << std::endl;
    return 0;
  }

  float resolution;
  if (argc > 2)
    resolution = atof (argv[2]);
  else
    resolution = 1.0;
  std::cerr << "Using octree resolution: " << resolution << std::endl;

  std::cerr << std::endl << "Computing.";

  CloudPtr cloud_target (new Cloud);
  pcl::io::loadPCDFile ("theatre_benchmark_target.pcd", *cloud_target);

  std::cerr << ".";

  CloudPtr cloud_in (new Cloud);
  pcl::io::loadPCDFile (argv[1], *cloud_in);

  std::cerr << ".";

  // Instantiate octree-based point cloud change detection classes
  pcl::octree::OctreePointCloudChangeDetector<PointType> octree_forward (resolution);
  octree_forward.setInputCloud (cloud_target);
  octree_forward.addPointsFromInputCloud ();
  octree_forward.switchBuffers ();
  octree_forward.setInputCloud (cloud_in);
  octree_forward.addPointsFromInputCloud ();

  std::cerr << ".";

  pcl::octree::OctreePointCloudChangeDetector<PointType> octree_backward (resolution);
  octree_backward.setInputCloud (cloud_in);
  octree_backward.addPointsFromInputCloud ();
  octree_backward.switchBuffers ();
  octree_backward.setInputCloud (cloud_target);
  octree_backward.addPointsFromInputCloud ();

  std::cerr << ".";

  // Get vector of point indices from octree voxels that do exist in second one but do not exist in first one
  std::vector<int> differences_forward;
  octree_forward.getPointIndicesFromNewVoxels (differences_forward);

  std::vector<int> differences_backward;
  octree_backward.getPointIndicesFromNewVoxels (differences_backward);

  std::cerr << ".";

  int noise_present = 0, noise_added = 0;
  for (size_t i = 0; i < differences_forward.size (); ++i)
  {
    if (checkPointInRange (cloud_in->points[differences_forward[i]], 500, 2000, 5000, 10000, -1825, 500))
      noise_present++;
    else if (checkPointInRange (cloud_in->points[differences_forward[i]], 0, 3000, 0, 1200, -1500, 2500))
      noise_present++;
    else if (checkPointInRange (cloud_in->points[differences_forward[i]], -3500, 3000, 8300, 10000, -1800, 6000))
      noise_present++;
    else if (checkPointInRange (cloud_in->points[differences_forward[i]], -3500, 3000, 0, 8300, 300, 6000))
      noise_present++;
    else
      noise_added++;
  }
  if (noise_present > 424122)
  {
    noise_added += noise_present - 424122;
    noise_present = 424122;
  }

  // Results
  std::cerr << std::endl << std::endl << "Target cloud: " << cloud_target->width << " x " << cloud_target->height << std::endl;
  std::cerr << "Input cloud: " << cloud_in->width << " x " << cloud_in->height << std::endl;
  std::cerr << "Difference: " << abs (cloud_target->points.size () - cloud_in->points.size ()) << " points" << std::endl;
  if (resolution <= 1.0)
  {
    std::cerr << std::endl << "Noise that was removed: " << 424122 - noise_present << " points" << std::endl;
    std::cerr << "Noise that was not removed: " << noise_present << " points" << std::endl;
    std::cerr << "Noise that was added: " << noise_added << " points" << std::endl;
    std::cerr << "Non-noise that was removed: " << differences_backward.size () << " points" << std::endl;
    std::cerr << std::endl << "Benchmark error result: " << noise_present / 424122 + 10 * noise_added / 4002050 + 10 * differences_backward.size () / 4002050;
    std::cerr << "   (0 is perfect, 1 is useless (can be higher than 1))" << std::endl;
  }
  return 0;
}
