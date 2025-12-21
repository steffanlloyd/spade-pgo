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
#include <geometry_msgs/Quaternion.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// Multi-drone swarm service and state message
#include "spade_pgo/ReinitSession.h"
#include "spade_pgo/PGOState.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "scancontext/Scancontext.h"

using namespace spade_pgo;

std::shared_ptr<PoseGraphManager> graph_manager;
std::shared_ptr<Visualizer> visualizer;
std::shared_ptr<LoopClosureManager> loop_closure_manager;
ros::Publisher state_publisher;

void process_pg()
{
    while(ros::ok())
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
    while(ros::ok())
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
    float vizmapFrequency = 0.1; // 0.1 means run onces every 10s
    ros::Rate rate(vizmapFrequency);
    while (ros::ok()) {
        rate.sleep();
        visualizer->publishMap();
    }
} // pointcloud_viz

// Service callback for reinitializing the PGO session for a new drone
bool reinitSessionCallback(
    spade_pgo::ReinitSession::Request& req,
    spade_pgo::ReinitSession::Response& res)
{
    (void)req;

    int next_kf_index = graph_manager->reinitializeSession();

    res.success = true;
    res.message = "Session reinitialized successfully";
    res.drone_id = graph_manager->getCurrentSessionId();
    res.next_keyframe_index = next_kf_index;

    ROS_INFO("ReinitSession service called. New drone_id: %d, next_keyframe_index: %d",
             res.drone_id, res.next_keyframe_index);

    return true;
}

// Publish PGO state at a given frequency
void process_state_publisher(void)
{
    float hz = 2.0;
    ros::Rate rate(hz);

    // Track session start index for current session
    int session_start_index = 0;

    while (ros::ok()) {
        rate.sleep();

        spade_pgo::PGOState state_msg;
        state_msg.header.stamp = ros::Time::now();
        state_msg.header.frame_id = "map";

        state_msg.current_drone_id = graph_manager->getCurrentSessionId();
        state_msg.num_keyframes = graph_manager->graphSize();
        state_msg.graph_initialized = graph_manager->graphInitialized();
        state_msg.gnss_initialized = graph_manager->isGNSSInitialized();

        state_msg.lc_candidate_queue_size = loop_closure_manager->getCandidateQueueSize();
        state_msg.lc_tested_count = loop_closure_manager->getTestedCandidatesCount();
        state_msg.lc_added_count = loop_closure_manager->getAddedLoopClosures().size();

        // Get session boundaries
        auto boundaries = graph_manager->getSessionBoundaries();
        for (const auto& boundary : boundaries) {
            state_msg.session_start_indices.push_back(boundary.first);
            state_msg.session_drone_ids.push_back(boundary.second);
        }

        // Calculate keyframes in current session
        if (!boundaries.empty()) {
            session_start_index = boundaries.back().first;
        }
        state_msg.num_keyframes_current_session = state_msg.num_keyframes - session_start_index;

        state_publisher.publish(state_msg);
    }
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserPGO");
	ros::NodeHandle nh;

    auto params = std::make_shared<PGOParams>();

    //ICP Params
    nh.param<std::string>("spade_pgo/icp/algorithm", params->icp.algorithm, "small_gicp"); // icp, gicp, ndt, small_gicp, or small_vgicp
    nh.param<int>("spade_pgo/icp/num_kf_accumulate_past", params->icp.num_kf_accumulate_past, 5); // Number of keyframes point cloud to be included in the submap for icp alignment (old keyframe)
    nh.param<int>("spade_pgo/icp/num_kf_accumulate_now", params->icp.num_kf_accumulate_now, 2); // Number of keyframes point cloud to be included in the submap for icp alignment (current keyframe)
    nh.param<double>("spade_pgo/icp/fitness_threshold", params->icp.fitness_threshold, 0.3); // ICP's loop fitness score threshold to accept closing a loop (i.e. registration was successful)
    nh.param<double>("spade_pgo/icp/min_inlier_ratio", params->icp.min_inlier_ratio, 0.1); // Minimum ratio of inliers to total correspondences (0 = disabled)
    nh.param<double>("spade_pgo/icp/voxel_size", params->icp.voxel_size, 0.2); // ICP's downsampling   
    nh.param<double>("spade_pgo/icp/max_correspondence_distance", params->icp.max_correspondence_distance, 2); // Maximum distance before points are ignored
    nh.param<int>("spade_pgo/icp/max_iterations", params->icp.max_iterations, 50); // Max ICP iterations
    nh.param<double>("spade_pgo/icp/max_height_above_ground", params->icp.max_height_above_ground, 0); // Filter points above this height relative to min z (0 = disabled)
    nh.param<double>("spade_pgo/icp/max_radius_from_keyframe", params->icp.max_radius_from_keyframe, 25); // Filter points beyond this distance from keyframe (0 = disabled)
    nh.param<bool>("spade_pgo/icp/save_pointclouds", params->icp.save_pointclouds, false); // Save pointclouds to file (for post-processing)
    nh.param<bool>("spade_pgo/icp/publish_pointclouds", params->icp.publish_pointclouds, false); // Publish point clouds (for debugging)

    // Graph parameters
	nh.param<double>("spade_pgo/graph/kf_gap_lin", params->graph.kf_gap_lin, 1.0); // Euclidean distance between keyframes (m)
	nh.param<double>("spade_pgo/graph/kf_gap_rot", params->graph.kf_gap_rot, 0.2); // Rotational distance between keyframes (rad)
    nh.param<double>("spade_pgo/graph/lio_noise_lin", params->graph.lio_noise_lin, 1e-3); // Std of noise from LIO (m)
    nh.param<double>("spade_pgo/graph/lio_noise_rot", params->graph.lio_noise_rot, 1e-2); // Std of noise from LIO (rad)
    nh.param<double>("spade_pgo/graph/prior_noise_lin", params->graph.prior_noise_lin, 5); // Prior noise from gps/imu initial pose estimate (m)
    nh.param<double>("spade_pgo/graph/prior_noise_rot", params->graph.prior_noise_rot, 0.2); // Prior noise from gps/imu initial pose estimate (rad)
    nh.param<bool>("spade_pgo/graph/use_gnss_altitude", params->graph.use_gnss_altitude, true); // Use GPS altitude
    nh.param<double>("spade_pgo/graph/gps_noise_threshold", params->graph.gps_noise_threshold, 4.0); // Covariance threshold for gps measurement to be taken into account (before applying scaling factor)
    nh.param<double>("spade_pgo/graph/gps_noise_scale", params->graph.gps_noise_scale, 1.0); // Scaling factor added to GPS variance. Value will be squared, so 2x value is 2x less certainty
    nh.param<double>("spade_pgo/graph/gps_noise_z_scale", params->graph.gps_noise_z_scale, 1e2); // Scaling factor added to GPS altitude variance. Value will be squared, so 2x value is 2x less certainty
    nh.param<double>("spade_pgo/graph/orientation_noise", params->graph.orientation_noise, 0.1);
    nh.param<double>("spade_pgo/graph/gnss_min_initialization_distance", params->graph.gnss_min_initialization_distance, 5); // Min distance travelled before GNSS will initialize (too small will destabilize map. Should be higher than covariance of GNSS)
    nh.param<double>("spade_pgo/graph/loop_closure_noise_scale", params->graph.loop_closure_noise_scale, 1); // Loop Noise scaling factor. Value will be squared

    // ScanControl parameters
    nh.param<bool>("spade_pgo/sc/enabled", params->sc.enabled, false);
    nh.param<double>("spade_pgo/sc/distance_threshold", params->sc.distance_threshold, 0.2);  
	nh.param<double>("spade_pgo/sc/max_radius",  params->sc.max_radius, 25.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor 
    nh.param<double>("spade_pgo/sc/voxel_size", params->sc.voxel_size, 0.4); // Scan Context point cloud downsampling
    
    // Near KF loop closure params
    nh.param<bool>("spade_pgo/near_kf/enabled", params->near_kf.enabled, false);
    nh.param<double>("spade_pgo/near_kf/distance_threshold", params->near_kf.distance_threshold, 10); // Distance that implies a potential match
    nh.param<double>("spade_pgo/near_kf/min_consecutive_kf_distance", params->near_kf.min_consecutive_kf_distance, 5); // Minimum distance between keyframes to be matched for loop closure (from another tested point)
    nh.param<int>("spade_pgo/near_kf/min_kf_seperation", params->near_kf.min_kf_seperation, 30); // Minimum number of keyframes between the two keyframes to be matched for loop closure (from another tested point)

    // Visualization parameters
	nh.param<double>("spade_pgo/visualize/voxel_size", params->visualize.voxel_size, 0.4); 

    // ROS parameters
    nh.param<std::string>("spade_pgo/ros/gps_topic", params->ros.gps_topic, ""); // Empty = disabled
    nh.param<std::string>("spade_pgo/ros/orientation_topic", params->ros.orientation_topic, "");  //  Empty = disabled. Must be in ENU frame
    nh.param<std::string>("spade_pgo/ros/orientation_msg_type", params->ros.orientation_msg_type, "odometry"); // "odometry" or "quaternion"
	nh.param<std::string>("spade_pgo/ros/pointcloud_topic", params->ros.pointcloud_topic, "/cloud_registered_body"); // Should be local frame (registered to the sensor body)
	nh.param<std::string>("spade_pgo/ros/lio_odometry_topic", params->ros.lio_odometry_topic, "/Odometry");
    nh.param<std::string>("spade_pgo/ros/save_directory", params->ros.save_directory, "/home/ros/save/pointclouds/");

    // Extrinsics: lidar to body transform as [roll, pitch, yaw] in radians
    std::vector<double> lidar_to_body_rpy;
    nh.param<std::vector<double>>("spade_pgo/extrinsics/lidar_to_body_rpy", lidar_to_body_rpy, {0.0, 0.0, 0.0});
    if (lidar_to_body_rpy.size() != 3) {
        ROS_WARN("lidar_to_body_rpy must have exactly 3 elements [roll, pitch, yaw]. Using identity.");
        lidar_to_body_rpy = {0.0, 0.0, 0.0};
    }
    params->extrinsics.T_body_lidar = Eigen::Isometry3d::Identity();
    params->extrinsics.T_body_lidar.linear() =
        (Eigen::AngleAxisd(lidar_to_body_rpy[2], Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(lidar_to_body_rpy[1], Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(lidar_to_body_rpy[0], Eigen::Vector3d::UnitX())).toRotationMatrix();
    ROS_INFO("Lidar-to-body extrinsics RPY [rad]: [%.3f, %.3f, %.3f]",
             lidar_to_body_rpy[0], lidar_to_body_rpy[1], lidar_to_body_rpy[2]);

    // Log which optional features are enabled
    ROS_INFO("GNSS integration: %s", params->useGNSS() ? "ENABLED" : "DISABLED");
    ROS_INFO("External orientation: %s", params->useExternalOrientation() ? "ENABLED" : "DISABLED");

    // Create save directories (fixed error check - don't fail on rm of non-existent dir)
    int ret1 = system((std::string("exec rm -r ") + params->ros.save_directory + " 2>/dev/null").c_str());
    (void)ret1;  // Ignore - directory may not exist
    int ret2 = system((std::string("mkdir -p ") + params->ros.save_directory).c_str());
    int ret3 = system((std::string("mkdir -p ") + params->ros.save_directory + "scans/").c_str());
    if (ret2 != 0 || ret3 != 0) {
        ROS_ERROR("Could not create the scan directory %s.", params->ros.save_directory.c_str());
    }

    graph_manager = std::make_shared<PoseGraphManager>(params);
    visualizer = std::make_shared<Visualizer>(graph_manager, nh, params);
    loop_closure_manager = std::make_shared<LoopClosureManager>(graph_manager, visualizer, params);
    visualizer->setLoopClosureManager(loop_closure_manager);
    graph_manager->setLoopClosureManager(loop_closure_manager);

    // Multi-drone swarm service and state publisher
    ros::ServiceServer reinit_service = nh.advertiseService("/spade_pgo/reinit_session", reinitSessionCallback);
    state_publisher = nh.advertise<spade_pgo::PGOState>("/spade_pgo/state", 10);
    ROS_INFO("Multi-drone swarm support enabled. Service: /spade_pgo/reinit_session, State: /spade_pgo/state");

    // Point cloud subscriber
    ros::Subscriber subscriber_pointcloud = nh.subscribe<sensor_msgs::PointCloud2>(
        params->ros.pointcloud_topic, 100, 
        std::function<void(const sensor_msgs::PointCloud2ConstPtr&)>(
            [](const sensor_msgs::PointCloud2ConstPtr &cloud) {
                graph_manager->data_buffer.pushPointCloud(cloud);
                visualizer->publishSensorCloud(cloud);
            }
        ));

    // Odometry subscriber
    ros::Subscriber subscriber_lio_odometry = nh.subscribe<nav_msgs::Odometry>(
        params->ros.lio_odometry_topic, 100, 
        std::function<void(const nav_msgs::Odometry::ConstPtr&)>(
            [](const nav_msgs::Odometry::ConstPtr &odom) {
                graph_manager->data_buffer.pushOdometry(odom);
            }
        ));

    // GNSS subscriber (only if topic is specified)
    ros::Subscriber subscriber_gnss;
    if (params->useGNSS()) {
        subscriber_gnss = nh.subscribe<sensor_msgs::NavSatFix>(
            params->ros.gps_topic, 100,
            std::function<void(const sensor_msgs::NavSatFix::ConstPtr&)>(
                [](const sensor_msgs::NavSatFix::ConstPtr &gps) {
                    graph_manager->data_buffer.pushGNSS(gps);
                }
            ));
        ROS_INFO("Subscribed to GNSS topic: %s", params->ros.gps_topic.c_str());
    }

    // Orientation subscriber (only if topic is specified)
    // Handles both Odometry and Quaternion message types via generic subscriber
    ros::Subscriber subscriber_orientation;
    if (params->useExternalOrientation()) {
        if (params->ros.orientation_msg_type == "odometry") {
            subscriber_orientation = nh.subscribe<nav_msgs::Odometry>(
                params->ros.orientation_topic, 100,
                std::function<void(const nav_msgs::Odometry::ConstPtr&)>(
                    [](const nav_msgs::Odometry::ConstPtr& odom) {
                        graph_manager->data_buffer.pushOrientation(odom);
                    }
                ));
        } else {
            subscriber_orientation = nh.subscribe<geometry_msgs::Quaternion>(
                params->ros.orientation_topic, 100,
                std::function<void(const geometry_msgs::Quaternion::ConstPtr&)>(
                    [](const geometry_msgs::Quaternion::ConstPtr& quat) {
                        graph_manager->data_buffer.pushOrientation(quat);
                    }
                ));
        }
        ROS_INFO("Subscribed to orientation topic: %s (type %s)", params->ros.orientation_topic.c_str(), params->ros.orientation_msg_type.c_str());
    }


	std::thread posegraph_slam {process_pg}; // pose graph construction
	std::thread lc_detection {process_lcd}; // loop closure detection
	std::thread icp_calculation {process_icp}; // loop constraint calculation via icp
	std::thread isam_update {process_isam}; // if you want to call less isam2 run (for saving redundant computations and no real-time visulization is required), uncommment this and comment all the above runisam2opt when node is added.
	std::thread viz_map {process_viz_map}; // visualization - map (low frequency because it is heavy)
	std::thread viz_path {process_viz_path}; // visualization - path (high frequency)
	std::thread state_pub {process_state_publisher}; // multi-drone state publisher

 	ros::spin();

	// Clean shutdown: wait for all threads to finish
	posegraph_slam.join();
	lc_detection.join();
	icp_calculation.join();
	isam_update.join();
	viz_map.join();
	viz_path.join();
	state_pub.join();

	return 0;
}