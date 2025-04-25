#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>

// spade_pgo includes
#include "spade_pgo/common.hpp"
#include "spade_pgo/DataBuffer.hpp"
#include "spade_pgo/geometry.hpp"
#include "spade_pgo/PGOParams.hpp"
#include "spade_pgo/PoseGraphManager.hpp"
#include "spade_pgo/ros_helpers.hpp"
#include "spade_pgo/Visualizer.hpp"
#include "spade_pgo/LoopClosureManager.hpp"

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/MagneticField.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "scancontext/Scancontext.h"

using namespace spade_pgo;

std::shared_ptr<PoseGraphManager> graph_manager;
std::shared_ptr<Visualizer> visualizer;
std::shared_ptr<LoopClosureManager> loop_closure_manager;

void process_pg()
{
    // SL: Start infinite loop
    while(1)
    {
        // While odometry buffer and laser scan buffer (full res) are not empty.
		while ( graph_manager->data_buffer.dataAvailable() )
        {
            // Get buffer values, or break if there isn't valid data
            auto data = graph_manager->data_buffer.popDataPoint();
            if(!data) break;

            graph_manager->processData( data.value() );
        }

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_pg

// Identify keyframes that have potential loop closures at a given rate
void process_lcd(void)
{
    float loopClosureFrequency = 0.4; // can change 
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
        rate.sleep();
        loop_closure_manager->updateCandidateQueue();
    }
} // process_lcd

// Compute exact graph factors from identified loop closure frames
void process_icp(void)
{
    while(1)
    {
		loop_closure_manager->processCandidateQueue();

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_icp

void process_viz_path(void)
{
    float hz = 10.0; 
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();
        visualizer->publishPath();
        visualizer->publishGPSPathAndMarkers();
        visualizer->publishLCMarkers();
    }
}

void process_isam(void)
{
    float hz = 1; 
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();

        graph_manager->optimizeGraph();
    }
}

// Publish visualization map at a given frequency
void process_viz_map(void)
{
    // SL (Q): vizmapFrequency should be a ROS parameter
    float vizmapFrequency = 0.1; // 0.1 means run onces every 10s
    ros::Rate rate(vizmapFrequency);
    while (ros::ok()) {
        rate.sleep();
        visualizer->publishMap();
    }
} // pointcloud_viz


int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserPGO");
	ros::NodeHandle nh;

    auto params = std::make_shared<PGOParams>();

    // ScanControl parameters
    nh.param<double>("sc_distance_threshold", params->sc.distance_threshold, 0.2);  
	nh.param<double>("sc_max_radius",  params->sc.max_radius, 25.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor 
    nh.param<double>("sc_voxel_size", params->sc.voxel_size, 0.4); // Scan Context point cloud downsampling

    //ICP Params
    nh.param<std::string>("icp_algorithm", params->icp.algorithm, "small_gicp"); // icp, gicp, ndt, small_gicp, or small_vgicp
    nh.param<int>("icp_num_kf_accumulate_past", params->icp.num_kf_accumulate_past, 5); // Number of keyframes point cloud to be included in the submap for icp alignment (old keyframe)
    nh.param<int>("icp_num_kf_accumulate_now", params->icp.num_kf_accumulate_now, 2); // Number of keyframes point cloud to be included in the submap for icp alignment (current keyframe)
    nh.param<double>("icp_fitness_threshold", params->icp.fitness_threshold, 0.3); // ICP's loop fitness score threshold to accept closing a loop (i.e. registration was successful)
    // average squared distances between matches
    nh.param<double>("icp_voxel_size", params->icp.voxel_size, 0.2); // ICP's downsampling   
    nh.param<double>("icp_max_correspondence_distance", params->icp.max_correspondence_distance, 2); // Maximum distance before points are ignored
    nh.param<bool>("icp_save_pointclouds", params->icp.save_pointclouds, false); // Save pointclouds to file (for post-processing)
    nh.param<bool>("icp_publish_pointclouds", params->icp.publish_pointclouds, false); // Publish point clouds (for debugging)

    // Graph parameters
	nh.param<double>("graph_kf_gap_lin", params->graph.kf_gap_lin, 1.0); // Euclidean distance between keyframes (m)
	nh.param<double>("graph_kf_gap_rot", params->graph.kf_gap_rot, 0.5); // Rotational distance between keyframes (rad)
    nh.param<double>("graph_lio_noise_lin", params->graph.lio_noise_lin, 1e-3); // Std of noise from LIO (m)
    nh.param<double>("graph_lio_noise_rot", params->graph.lio_noise_rot, 1e-2); // Std of noise from LIO (rad)
    nh.param<double>("graph_prior_noise_lin", params->graph.prior_noise_lin, 5); // Prior noise from gps/imu initial pose estimate (m)
    nh.param<double>("graph_prior_noise_rot", params->graph.prior_noise_rot, 0.2); // Prior noise from gps/imu initial pose estimate (rad)
    nh.param<bool>("graph_use_gnss", params->graph.use_gnss, true); // Use GPS or not
    nh.param<bool>("graph_use_gnss_altitude", params->graph.use_gnss_altitude, true); // Use GPS altitude
    nh.param<double>("graph_gps_noise_threshold", params->graph.gps_noise_threshold, 4.0); // Covariance threshold for gps measurement to be taken into account (before applying scaling factor)
    nh.param<double>("graph_gps_noise_scale", params->graph.gps_noise_scale, 1.0); // Scaling factor added to GPS variance. Value will be squared, so 2x value is 2x less certainty
    nh.param<double>("graph_gps_noise_z_scale", params->graph.gps_noise_z_scale, 1e2); // Scaling factor added to GPS altitude variance. Value will be squared, so 2x value is 2x less certainty
    nh.param<double>("graph_gnss_min_initialization_distance", params->graph.gnss_min_initialization_distance, 5); // Min distance travelled before GNSS will initialize (too small will destabilize map. Should be higher than covariance of GNSS)
    nh.param<double>("graph_loop_closure_noise_scale", params->graph.loop_closure_noise_scale, 1); // Loop Noise scaling factor. Value will be squared
    nh.param<int>("graph_orientation_calibration_size", params->graph.orientation_calibration_size, 20); // Number of samples from the imu and heading to calculate the initial pose
    nh.param<bool>("graph_use_orientation_calibration", params->graph.use_orientation_calibration, true); // Whether or not to use the orientation calibration procedue. If yes, must specify an IMU and heading stream.

    // Loop closure params
    nh.param<bool>("loop_closure_use_scancontrol", params->loop_closure.use_scancontrol, false);
    nh.param<bool>("loop_closure_use_near_kf", params->loop_closure.use_near_kf, false);
    nh.param<double>("loop_closure_kf_distance", params->loop_closure.kf_distance, 5.0); // Distance between keyframes to be considered for loop closure (m)

    // Near KF loop closure commands
    nh.param<double>("near_kf_distance_threshold", params->near_kf.distance_threshold, 10); // Distance that implies a potential match
    nh.param<double>("near_kf_min_consecutive_kf_distance", params->near_kf.min_consecutive_kf_distance, 5); // Minimum distance between keyframes to be matched for loop closure (from another tested point)
    nh.param<int>("near_kf_min_kf_seperation", params->near_kf.min_kf_seperation, 30); // Minimum number of keyframes between the two keyframes to be matched for loop closure (from another tested point)

    // Visualization parameters
	nh.param<double>("visualize_voxel_size", params->visualize.voxel_size, 0.4); // pose assignment every k frames 

    // ROS parameters
    nh.param<std::string>("ros_gps_topic", params->ros.gps_topic, "/mavros/global_position/global");
	nh.param<std::string>("ros_pointcloud_topic", params->ros.pointcloud_topic, "/cloud_registered_body"); // Should be local frame (registered to the sensor body)
	nh.param<std::string>("ros_odometry_topic", params->ros.odometry_topic, "/Odometry");
    nh.param<std::string>("ros_imu_topic", params->ros.imu_topic, "/mavros/imu/data");
    nh.param<std::string>("ros_heading_topic", params->ros.heading_topic, "/mavros/global_position/compass_hdg");
    nh.param<std::string>("ros_save_directory", params->ros.save_directory, "/");

    int ret1 = system((std::string("exec rm -r ") + params->ros.save_directory).c_str());
    int ret2 = system((std::string("mkdir -p ") + params->ros.save_directory).c_str());
    int ret3 = system((std::string("mkdir -p ") + params->ros.save_directory  + "scans/").c_str());
    if (ret1!=0 || ret2!=0 || ret3!=0) ROS_ERROR("Could not reset and create the scan directory %s.", params->ros.save_directory.c_str());

    graph_manager = std::make_shared<PoseGraphManager>(params);
    visualizer = std::make_shared<Visualizer>(graph_manager, nh, params);
    loop_closure_manager = std::make_shared<LoopClosureManager>(graph_manager, visualizer, params);
    visualizer->setLoopClosureManager(loop_closure_manager);
    graph_manager->setLoopClosureManager(loop_closure_manager);

	ros::Subscriber subscriber_pointcloud = nh.subscribe<sensor_msgs::PointCloud2>(
        params->ros.pointcloud_topic, 100, 
        std::function<void(const sensor_msgs::PointCloud2ConstPtr&)>(
            [](const sensor_msgs::PointCloud2ConstPtr &cloud) {
                graph_manager->data_buffer.pushPointCloud(cloud);
                visualizer->publishSensorCloud(cloud);
            }
        ));
	ros::Subscriber subscriber_lio_odometry = nh.subscribe<nav_msgs::Odometry>(
        params->ros.odometry_topic, 100, 
        std::function<void(const nav_msgs::Odometry::ConstPtr&)>(
            [](const nav_msgs::Odometry::ConstPtr &odom) {
                graph_manager->data_buffer.pushOdometry(odom);
            }
        ));
    ros::Subscriber subscriber_gnss;
    if(params->graph.use_gnss){
        subscriber_gnss = nh.subscribe<sensor_msgs::NavSatFix>(
            params->ros.gps_topic, 100,
            std::function<void(const sensor_msgs::NavSatFix::ConstPtr&)>(
                [&params](const sensor_msgs::NavSatFix::ConstPtr &gps) {
                    if(params->graph.use_gnss){
                        graph_manager->data_buffer.pushGNSS(gps);
                    }
                }
            ));
    }
    // To do: Only make these subscribers if doing calibration
    ros::Subscriber subscriber_heading = nh.subscribe<std_msgs::Float64>(
        params->ros.heading_topic, 100,
        std::function<void(const std_msgs::Float64::ConstPtr&)>(
            [](const std_msgs::Float64::ConstPtr& msg) {
                graph_manager->orienter->pushHeading(msg);
            }
        ));
    ros::Subscriber subscriber_imu = nh.subscribe<sensor_msgs::Imu>(
        params->ros.imu_topic, 100,
        std::function<void(const sensor_msgs::Imu::ConstPtr&)>(
            [](const sensor_msgs::Imu::ConstPtr& msg) {
                graph_manager->orienter->pushIMU(msg);
            }
        ));	

	std::thread posegraph_slam {process_pg}; // pose graph construction
	std::thread lc_detection {process_lcd}; // loop closure detection 
	std::thread icp_calculation {process_icp}; // loop constraint calculation via icp 
    // SL (Q): Should define ROS parameters to decide whether to run isam2 as you described below, or the other option.
    // I'm not 100% sure I understand what you mean by "uncomment this and comment all the above runisam2opt when node is added".
	std::thread isam_update {process_isam}; // if you want to call less isam2 run (for saving redundant computations and no real-time visulization is required), uncommment this and comment all the above runisam2opt when node is added. 

    // SL (Q): Visualization update frequency should be a ros parameter. And then we can add logic that if that
    // parameter is negative (e.g. -1), that means to not run it at all.
	std::thread viz_map {process_viz_map}; // visualization - map (low frequency because it is heavy)
	std::thread viz_path {process_viz_path}; // visualization - path (high frequency)

 	ros::spin();

	return 0;
}