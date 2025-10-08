#include <iostream>
#include <string>
#include <thread>
#include <chrono>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>


int main(int argc, char** argv)
{
    // Default filenames; optionally pass them via command-line:
    //   ./icp_debug source_cloud.pcd target_cloud.pcd
    std::string source_filename = "/home/ros/save/pointclouds/loop_closure_pcd/current_kf_2437-91.pcd";
    std::string target_filename = "/home/ros/save/pointclouds/loop_closure_pcd/loop_kf_2437-91.pcd";
    if (argc >= 3) {
        source_filename = argv[1];
        target_filename = argv[2];
    } else {
        std::cout << "Using default filenames:\n  Source: " << source_filename 
                  << "\n  Target: " << target_filename << std::endl;
    }

    // Define the point cloud type (change if needed)
    using PointType = pcl::PointXYZI;
    pcl::PointCloud<PointType>::Ptr source_cloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr target_cloud(new pcl::PointCloud<PointType>());

    // Load the PCD files
    if (pcl::io::loadPCDFile<PointType>(source_filename, *source_cloud) == -1)
    {
        PCL_ERROR("Couldn't read source file\n");
        return -1;
    }
    if (pcl::io::loadPCDFile<PointType>(target_filename, *target_cloud) == -1)
    {
        PCL_ERROR("Couldn't read target file\n");
        return -1;
    }
    std::cout << "Loaded " << source_cloud->points.size() << " points from " << source_filename << std::endl;
    std::cout << "Loaded " << target_cloud->points.size() << " points from " << target_filename << std::endl;
    
    pcl::PointCloud<PointType> aligned_cloud;

    
    
    // ------------------------
    //  ICP Registration
    // ------------------------
    // pcl::IterativeClosestPoint<PointType, PointType> icp;
    pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(1);   // Smaller threshold for fine alignment
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(10);

    icp.setInputSource(source_cloud);
    icp.setInputTarget(target_cloud);

    icp.align(aligned_cloud);

    if (!icp.hasConverged())
    {
        std::cout << "ICP did not converge." << std::endl;
        return -1;
    }
    else
    {
        std::cout << "ICP converged." << std::endl;
        std::cout << "Fitness score: " << icp.getFitnessScore() << std::endl;
        std::cout << "Transformation matrix:" << std::endl << icp.getFinalTransformation() << std::endl;
    }
    
    
    // ------------------------
    //  NDT Registration
    // ------------------------
    // pcl::NormalDistributionsTransform<PointType, PointType> ndt;
    // ndt.setResolution(.6); // Adjust resolution (in meters) as needed
    // ndt.setMaxCorrespondenceDistance(2);   // Maximum correspondence distance
    // ndt.setMaximumIterations(100);
    // ndt.setTransformationEpsilon(1e-7);
    // ndt.setEuclideanFitnessEpsilon(1e-7);

    // ndt.setInputSource(source_cloud);
    // ndt.setInputTarget(target_cloud);

    // pcl::PointCloud<PointType>::Ptr ndt_result(new pcl::PointCloud<PointType>());
    // // Use identity as initial guess (or provide a prior estimate if available)
    // Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
    // ndt.align(aligned_cloud, initial_guess);
    
    // if (ndt.hasConverged() == false) {
    //     std::cout << "ICP did not converge." << std::endl;
    //     return -1;
    // } else {
    //     std::cout << "ICP converged." << std::endl;
    //     std::cout << "Fitness score: " << ndt.getFitnessScore() << std::endl;
    //     std::cout << "Transformation matrix:" << std::endl << ndt.getFinalTransformation() << std::endl;
    // }

    // Visualizer 1: Initial Clouds
    pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Initial Registration"));
    viewer1->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<PointType> target_color(target_cloud, 255, 0, 0);
    viewer1->addPointCloud<PointType>(target_cloud, target_color, "target cloud");
    pcl::visualization::PointCloudColorHandlerCustom<PointType> raw_source_color(source_cloud, 0, 0, 255);
    viewer1->addPointCloud<PointType>(source_cloud, raw_source_color, "raw source cloud");

    // Visualizer 2: Aligned Clouds
    pcl::visualization::PCLVisualizer::Ptr viewer2(new pcl::visualization::PCLVisualizer("Aligned Registration"));
    viewer2->setBackgroundColor(0, 0, 0);
    viewer2->addPointCloud<PointType>(target_cloud, target_color, "target cloud");
    pcl::visualization::PointCloudColorHandlerCustom<PointType> aligned_color(aligned_cloud.makeShared(), 0, 255, 0);
    viewer2->addPointCloud<PointType>(aligned_cloud.makeShared(), aligned_color, "aligned cloud");

    // Run both visualizers
    while (!viewer1->wasStopped() && !viewer2->wasStopped()) {
        viewer1->spinOnce(100);
        viewer2->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}


//     // Visualize the registration result using PCLVisualizer
//     pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("ICP Debug"));
//     viewer->setBackgroundColor(0, 0, 0);

//     // Add the target cloud (displayed in red)
//     pcl::visualization::PointCloudColorHandlerCustom<PointType> target_color(target_cloud, 255, 0, 0);
//     viewer->addPointCloud<PointType>(target_cloud, target_color, "target cloud");
//     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target cloud");
    
//     // Add the initial (raw) source cloud in blue
//     pcl::visualization::PointCloudColorHandlerCustom<PointType> raw_source_color(source_cloud, 0, 0, 255);
//     viewer->addPointCloud<PointType>(source_cloud, raw_source_color, "raw source cloud");
//     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "raw source cloud");


//     // Add the aligned source cloud (displayed in green)
//     pcl::visualization::PointCloudColorHandlerCustom<PointType> source_color(aligned_cloud.makeShared(), 0, 255, 0);
//     viewer->addPointCloud<PointType>(aligned_cloud.makeShared(), source_color, "aligned source cloud");
//     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "aligned source cloud");

//     // Optionally, add coordinate axes
//     viewer->addCoordinateSystem(1.0);

//     // Main loop: keep the visualizer window open until the user closes it
//     while (!viewer->wasStopped())
//     {
//         viewer->spinOnce(100);
//         std::this_thread::sleep_for(std::chrono::milliseconds(100));
//     }

//     return 0;
// }
