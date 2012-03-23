#include <iostream>
#include <stdio.h>
#include <pcl/point_types.h>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include "statistical.h"
#include "regionGrowing.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/common/time.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d.h>

#include "opencv_cloud.h"
#include "classification.h"

typedef pcl::PointXYZI PointType;

bool filterIt(int argc, char **argv,pcl::PointCloud<PointType>::Ptr cloud, pcl::PointCloud<PointType>::Ptr cloud_filtered) {

    pcl::ScopeTime time("performance");
    time.reset();

    if (strcmp(argv[2], "-r") == 0) {
        if (argc != 5) {
            std::cout << "All the parameters are not correctly set for the radius removal filter" << std::endl;
            return 0;
        }
        pcl::RadiusOutlierRemoval<PointType> outrem;
        // build the filter
        outrem.setInputCloud(cloud);
        outrem.setRadiusSearch(atof(argv[3]));
        outrem.setMinNeighborsInRadius (atof(argv[4]));
        // apply filter
        outrem.filter (*cloud_filtered);
    }
    else if (strcmp(argv[2], "-c") == 0) {
        if (argc != 6) {
            std::cout << "All the parameters are not correctly set for the conditional filter" << std::endl;
            return 0;
        }
        // build the condition
        pcl::ConditionAnd<PointType>::Ptr range_cond (new  pcl::ConditionAnd<PointType> ());
        range_cond->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new
                                   pcl::FieldComparison<PointType> (argv[5], pcl::ComparisonOps::GT, atof ( argv[3] ))));
        range_cond->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new
                                   pcl::FieldComparison<PointType> (argv[5], pcl::ComparisonOps::LT, atof(argv[4]) )));
        // build the filter
        pcl::ConditionalRemoval<PointType> condrem (range_cond);
        condrem.setInputCloud (cloud);
        condrem.setKeepOrganized(false);
        // apply filter
        condrem.filter (*cloud_filtered);
    }
    else if (strcmp(argv[2], "-s") == 0) {
        if (argc != 5) {
            std::cout << "All the parameters are not correctly set for the statistical filter" << std::endl;
            return 0;
        }
        // Create the filtering object
        pcl::StatisticalOutlierRemoval<PointType> sor;
        sor.setInputCloud (cloud);
        sor.setMeanK (atof(argv[3]));
        sor.setStddevMulThresh (atof(argv[4]));
        sor.filter (*cloud_filtered);
    }
    else if (strcmp(argv[2], "-v") == 0) {
        // Create the filtering object
        pcl::VoxelGrid<PointType> sor;
        sor.setDownsampleAllData(true);
        sor.setInputCloud (cloud);
        sor.setLeafSize (atof(argv[3]), atof(argv[3]), atof(argv[3]));
        sor.filter (*cloud_filtered);
    }
    else if (strcmp(argv[2], "-av") == 0) {
        // Create the filtering object
        pcl::ApproximateVoxelGrid<PointType> sor;
        sor.setDownsampleAllData(true);
        sor.setInputCloud (cloud);
        sor.setLeafSize (atof(argv[3]), atof(argv[3]), atof(argv[3]));
        sor.filter (*cloud_filtered);
    }
    else if (strcmp(argv[2], "-rg") == 0) {
        // Create the filtering object
        std::vector<pcl::IndicesPtr> clusteredIndices;
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
        pcl::RegionGrowing<PointType> sor;
        //sor.setDownsampleAllData(true);
        sor.setInputCloud (cloud);
        sor.setGrowingDistance(atof(argv[3]));
        //sor.setLeafSize (atof(argv[3]), atof(argv[3]), atof(argv[3]));
        
        if (atof(argv[4]) > 0){
            sor.setEpsAngle(atof(argv[4]));
            pcl::NormalEstimation<PointType, pcl::Normal> ne;
            ne.setInputCloud (cloud);
            pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());
            ne.setSearchMethod (tree);
            ne.setRadiusSearch (atof(argv[3]));
            ne.compute (*cloud_normals);
	    sor.setNormals(cloud_normals);
	    cout << "normals estimed" << endl;
        }
        
	sor.cluster (&clusteredIndices);
        cout << "clusters? " <<clusteredIndices.size()<<endl;
	
	classification<PointType> classif(cloud, clusteredIndices);
        // pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
        // tree->setInputCloud (cloud);

        //extractEuclideanClusters (
//         const PointCloud<PointT> &cloud, const boost::shared_ptr<search::Search<PointT> > &tree,
//         float tolerance, std::vector<PointIndices> &clusters,
//         unsigned int min_pts_per_cluster = 1, unsigned int max_pts_per_cluster = (std::numeric_limits<int>::max) ());

// 	 boost::shared_ptr<pcl::search::KdTree<PointType> > tree (new pcl::search::KdTree<PointType>);
// 	 tree->setInputCloud (cloud);
// 	 std::vector<pcl::PointIndices> clusters;
//          pcl::extractEuclideanClusters<PointType>(cloud, tree, 10, clusters, 100, 200);
//         pcl::EuclideanClusterExtraction<PointType> ec;
//         ec.setClusterTolerance ( atof(argv[3]) ); // 2cm
//         ec.setSearchMethod (tree);
//         ec.setInputCloud (cloud);
//         ec.extract (clusteredIndices);
//        cout << "clusters? " <<clusteredIndices.size()<<endl;
        pcl::PointCloud<PointType>::Ptr buff_cloud (new pcl::PointCloud<PointType>);

        for (int i=0; i<clusteredIndices.size(); i++) {
            //cout << "ok 2" << endl;
            if (clusteredIndices.operator[](i)->size() < 2300 || clusteredIndices.operator[](i)->size() > 2500) {
//                 cout << "ok 3 " << i << endl;
                //cout << clusteredIndices[i]->size() << endl;
                pcl::copyPointCloud(*cloud, clusteredIndices[i].operator*(), *buff_cloud);
                cloud_filtered->operator+=(*buff_cloud);
                buff_cloud->clear();
            }

        }
        if(system("mkdir clusters"))
	  cout << "creating directory \"clusters\""<< endl;
        saveclusters(clusteredIndices, cloud, "clusters/");
        std::stringstream command;
        command << "pcd_viewer ";
        for (int i =0 ; i< clusteredIndices.size(); i++)
	 // if (clusteredIndices[i].indices.size() > 2300 && clusteredIndices[i].indices.size() < 2500)
            command << "clusters/" << i << ".pcd ";
        if(system (command.str().data()))
	  cout << "visualizing clusters" << endl;
        FILE *fp;
        fp = fopen ( "command.sh", "wt" ) ;
        fprintf ( fp, " %s\n", command.str().data() );
        fclose(fp);
	
	
    }
    else {
        std::cerr << "please specify the point cloud and the command line arg '-r' or '-c' or '-s' or '-v' or '-av' or 'rg' + param" << std::endl;
        exit(0);
    }

    time.getTime();
    return 1;
}

int
main (int argc, char** argv)
{

    if (argc < 4)
    {
        std::cerr << "please specify the point cloud and the command line arg '-r' or '-c' or '-s' or '-v' or '-av' or '-rg' + param\n" << std::endl;
        std::cerr << "radius filter: param ==> ray + min_neighbours " << std::endl;
        std::cerr << "    example: " << argv[0] << " cloud.pcd -r 100 10 \n" << std::endl;
        std::cerr << "codnitional filter: param ==> min_dist + max_dist along the axis" << std::endl;
        std::cerr << "    example: " << argv[0] << " cloud.pcd -c 0 1000 x\n" << std::endl;
        std::cerr << "statistical filter: param ==> num_of_neigh + std_dev" << std::endl;
        std::cerr << "    example: " << argv[0] << " cloud.pcd -s 50 1\n" << std::endl;
        std::cerr << "voxel grid downsampling: param ==> leaf_size" << std::endl;
        std::cerr << "    example: " << argv[0] << " cloud.pcd -v 10\n" << std::endl;
        std::cerr << "voxel grid (approximated) downsampling filter: param ==> leafsize" << std::endl;
        std::cerr << "    example: " << argv[0] << " cloud.pcd -av 10\n" << std::endl;
        std::cerr << "region growing: param ==> point_dist" << std::endl;
        std::cerr << "    example: " << argv[0] << " cloud.pcd -rg 10\n" << std::endl;
        exit(0);
    }
    pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud_filtered (new pcl::PointCloud<PointType>);
    if (pcl::io::loadPCDFile (argv[1], *cloud))
        return 0;
    cloud->height = 1;
    cloud->width = cloud->size();
    cloud->is_dense=0;

    do {
        pcl::PointCloud<PointType>::Ptr cloud_filtered (new pcl::PointCloud<PointType>);

        std::stringstream string;
        size_t begin;
        std::string buff;
        string << "output/";
        //if (system("mkdir output")) int a=0;

        buff.assign(argv[1]);
        // Check if the filename is a path
        begin=buff.find_last_of("/\\");
        if ( begin!=std::string::npos )
            buff=buff.substr(begin+1);

        begin=buff.find(".pcd");
        if ( begin!=std::string::npos )
            buff.assign ( buff,0,begin );
        else {
            std::cout << "No valid .pcd loaded" << std::endl;
            return 0;
        }
        string << buff;

        buff.assign(argv[2]);
        begin=buff.find("-");
        if ( begin!=std::string::npos )
            buff.assign ( buff, begin+1, buff.length() );
        else {
            std::cout << "No valid method defined" << std::endl;
            return 0;
        }
        string << "_" << buff;
        buff.clear();

        if (argc > 3)
            buff.assign(argv[3]);
        if ( buff.length() > 0 )
            string << "_" << buff;
        buff.clear();

        if (argc > 4)
            buff.assign(argv[4]);
        if ( buff.length() > 0 )
            string << "_" << buff;
        buff.clear();

        if (argc > 5)
            buff.assign(argv[5]);
        if ( buff.length() > 0 )
            string << "_" << buff;

        string << ".pcd";

        if (!filterIt(argc, argv, cloud, cloud_filtered))
            return 0;

        //cloud_filtered->is_dense = 0;


        // If ordered, remove the NaN pointsd
        if (0) {
            std::cout << "Cloud is organized and NaN points are now removed" << std::endl;
            std::vector<int> unused;
            cloud_filtered->is_dense = 0;
            pcl::removeNaNFromPointCloud (*cloud_filtered, *cloud_filtered, unused);
        }

        //pcl::io::savePCDFile<PointType>(string.str(), cloud_filtered.operator*());
        std::cerr << "Cloud points before filtering: " << cloud->points.size() << std::endl;
        std::cerr << "Cloud points after filtering: " << cloud_filtered->points.size() << std::endl;
        pcl::io::savePCDFileBinary<PointType>(string.str(), *cloud_filtered);
        std::cout << "Resulting cloud saved in " << string.str() << std::endl;

        std::stringstream command;
        command << "pcd_viewer_int " << argv[1] << " " << string.str();
        std::cout << "Displaying the result... " << argv[2] << " "<<argv[3];
        if (!system(command.str().data()))
            std::cout << "stop" << std::endl << std::endl;
        std::cout << "repeat? [y/n]" << std::endl;
        char c;
        cloud->clear();
        cloud = cloud_filtered;
        c=getchar();
        getchar();
        if (c != 'y')
            break;
    } while (1);
    return (0);
}