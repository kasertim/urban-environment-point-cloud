#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/bilateral.h>

#include <pcl/filters/passthrough.h>
#include <pcl/common/time.h>

typedef pcl::PointXYZI PointType;
typedef pcl::PointCloud<PointType> Cloud;
typedef Cloud::Ptr CloudPtr;

int
main (int argc, char** argv)
{
    CloudPtr cloud_in (new Cloud);
    CloudPtr cloud_out (new Cloud);
    pcl::ScopeTime time("performance");
    float endTime;
    pcl::io::loadPCDFile (argv[1], *cloud_in);
    std::cout << "Input size: " << cloud_in->width << " by " << cloud_in->height << std::endl;

///////////////////////////////////////////////////////////////////////////////////////////
//
 pcl::PassThrough<PointType> pass;
 pass.setInputCloud (cloud_in);
 //pass.setKeepOrganized(true);

 pass.setFilterFieldName ("x");
 pass.setFilterLimits (1300, 2200.0);
 pass.filter (*cloud_in);

 pass.setFilterFieldName ("y");
 pass.setFilterLimits (4000, 10000);
 pass.filter (*cloud_in);

 pass.setFilterFieldName ("z");
 pass.setFilterLimits (-2000, 500.0);
 pass.filter (*cloud_out);
//
///////////////////////////////////////////////////////////////////////////////////////////
//
//  std::vector<int> unused;
//  pcl::removeNaNFromPointCloud (*cloud_in, *cloud_in, unused);
//
///////////////////////////////////////////////////////////////////////////////////////////
//
//  pcl::search::Search<PointType>::Ptr searcher (new pcl::search::KdTree<PointType> (false));
//  searcher->setInputCloud (cloud_in);
//  PointType point;
//  point.x = -10000;
//  point.y = 0;
//  point.z = 0;
//  std::vector<int> k_indices;
//  std::vector<float> unused2;
//  //searcher->radiusSearch (point, 100, k_indices, unused2);
//  searcher->nearestKSearch (point, 500000, k_indices, unused2);
//  cloud_out->width = k_indices.size ();
//  cloud_out->height = 1;
//  cloud_out->is_dense = false;
//  cloud_out->points.resize (cloud_out->width);
//  for (size_t i = 0; i < cloud_out->points.size (); ++i)
//  {
//    cloud_out->points[i] = cloud_in->points[k_indices[i]];
//  }
//
///////////////////////////////////////////////////////////////////////////////////////////
//
//  cloud_out->width = cloud_in->width / 10;
//  cloud_out->height = 1;
//  cloud_out->is_dense = false;
//  cloud_out->points.resize (cloud_out->width);
//  for (size_t i = 0; i < cloud_out->points.size (); ++i)
//  {
//    cloud_out->points[i] = cloud_in->points[i * 10];
//  }
//
///////////////////////////////////////////////////////////////////////////////////////////

//     pcl::BilateralFilter<PointType> filter;
//     filter.setInputCloud (cloud_in);
//     pcl::octree::OctreeLeafDataTVector<int> leafT;
//     pcl::search::Search<PointType>::Ptr searcher (new pcl::search::Octree
//             < PointType,
//             pcl::octree::OctreeLeafDataTVector<int> ,
//             pcl::octree::OctreeBase<int, pcl::octree::OctreeLeafDataTVector<int> >
//             > (500) );
//     //pcl::search::Octree <PointType> ocTreeSearch(1);
//     filter.setSearchMethod (searcher);
//     double sigma_s, sigma_r;
// 
//     filter.setHalfSize (500);
//     time.reset();
//     filter.filter (*cloud_out);
//     time.getTime();
//     std::cout << "Benchmark Bilateral filer using: kdtree searching, sigma_s = 50, sigma_r = default" << std::endl;
// std::cout << "Results: clocks = " << end - start << ", clockspersec = " << CLOCKS_PER_SEC << std::endl;

//  std::ofstream filestream;
//  filestream.open ("timer_results.txt");
//  char filename[50];
//  for (sigma_s = 1000; sigma_s <= 1000; sigma_s *= 10)
//    for (sigma_r = 0.001; sigma_r <= 1000; sigma_r *= 10)
//    {
//      std::cout << "sigma_s = " << sigma_s << ", sigma_r = " << sigma_r << " clockspersec = " << CLOCKS_PER_SEC
//          << std::endl;
//      filter.setHalfSize (sigma_s);
//      filter.setStdDev (sigma_r);
//      start = clock ();
//      filter.filter (*cloud_out);
//      end = clock ();
//      sprintf (filename, "bruteforce-s%g-r%g.pcd", sigma_s, sigma_r);
//      pcl::io::savePCDFileBinary (filename, *cloud_out);
//      filestream << "sigma_s = " << sigma_s << ", sigma_r = " << sigma_r << " time = " << end - start
//          << " clockspersec = " << CLOCKS_PER_SEC << std::endl;
//    }
//  filestream.close ();

//  std::cout << "Output size: " << cloud_out->width << " by " << cloud_out->height << std::endl;
 pcl::io::savePCDFileBinary ("cropped_ghosts.pcd", *cloud_out);
    std::cout << std::endl << "Goodbye World" << std::endl << std::endl;
    return (0);
}
