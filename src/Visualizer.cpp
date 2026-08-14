#include "spade_pgo/Visualizer.hpp"
#include "spade_pgo/ros_helpers.hpp"
#include "spade_pgo/common.hpp"
#include "spade_pgo/geometry.hpp"
#include "spade_pgo/LoopClosureManager.hpp"

#include <ros/ros.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>

namespace spade_pgo {

Visualizer::Visualizer(
    std::shared_ptr<PoseGraphManager> graph_manager,
    ros::NodeHandle& nh, 
    std::shared_ptr<PGOParams> params)
    : params(params), graph_manager_(graph_manager), nh_(nh)
{
    // Setup publishers
    this->publisher_path_         = nh_.advertise<nav_msgs::Path>("/pgo/path", 100);
    this->publisher_odometry_     = nh_.advertise<nav_msgs::Odometry>("/pgo/odom", 100);
    this->publisher_lc_markers_   = nh_.advertise<visualization_msgs::MarkerArray>("pgo/loop_closure_markers", 10);
    this->publisher_gps_markers_  = nh_.advertise<visualization_msgs::MarkerArray>("pgo/gps_markers", 10);
    this->publisher_gps_path_     = nh_.advertise<nav_msgs::Path>("/pgo/gps_path", 100);
    this->publisher_map_          = nh_.advertise<sensor_msgs::PointCloud2>("/pgo/map", 100);
    this->publisher_sensor_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/pgo/sensor_cloud", 100);
    this->publisher_lc_cloud_curr_= nh_.advertise<sensor_msgs::PointCloud2>("/pgo/lc_cloud_curr", 100);
    this->publisher_lc_cloud_prev_= nh_.advertise<sensor_msgs::PointCloud2>("/pgo/lc_cloud_prev", 100);

    // Set up voxelizer
    this->voxelizer_map_.setLeafSize(params->visualize.voxel_size, params->visualize.voxel_size, params->visualize.voxel_size);

    // Make save directory
    this->lc_pcd_directory_ = params->ros.save_directory + "lc_pcd/";
    if(this->params->icp.save_pointclouds){
        int ret1 = system((std::string("exec rm -r ") + this->lc_pcd_directory_).c_str());
        int ret2 = system((std::string("mkdir -p ") + this->lc_pcd_directory_).c_str());
        if (ret1!=0 || ret2!=0) ROS_ERROR("Could not reset and create the lc pcd directory %s.", this->lc_pcd_directory_.c_str());
    }

}

Visualizer::~Visualizer() {}

// void Visualizer::publishLoopClosureMarkers()
// {
//     visualization_msgs::MarkerArray markerArray;
//     auto kf_updated = graph_manager_->getUpdatedKFPoses();
    
//     // Iterate over global loop closures (assumed available in loopClosuresAdded)
//     for (size_t i = 0; i < loopClosuresAdded.size(); ++i) {
//         int id1 = loopClosuresAdded[i].id1;
//         int id2 = loopClosuresAdded[i].id2;
//         if (id1 < 0 || id1 >= static_cast<int>(kf_updated.size()) ||
//             id2 < 0 || id2 >= static_cast<int>(kf_updated.size()))
//         {
//             continue;
//         }
        
//         geometry_msgs::Point p1 = isometryToPoint(kf_updated[id1]);
//         geometry_msgs::Point p2 = isometryToPoint(kf_updated[id2]);
//         std::vector<geometry_msgs::Point> ptsA = { p1 };
//         std::vector<geometry_msgs::Point> ptsB = { p2 };
        
//         std_msgs::ColorRGBA color;
//         color.r = 1.0; color.g = 0.0; color.b = 0.0; color.a = 0.4;
        
//         visualization_msgs::Marker marker = createMarkerLines(ptsA, ptsB, color, "camera_init", i, 0.2);
//         markerArray.markers.push_back(marker);
//     }
//     publisher_lc_markers_.publish(markerArray);
// }

/**
 * @brief Publishes GPS path and corresponding blue markers.
 */
void Visualizer::publishGPSPathAndMarkers()
{
    if(!this->graph_manager_->graphInitialized() || this->graph_manager_->updatedKFIndex()<1) return;

    // Get points
    auto gnss_points = this->graph_manager_->getGNSSPoints();
    auto updated_kf_poses = this->graph_manager_->getUpdatedKFPoses();
    auto kf_timestamps = this->graph_manager_->getKFTimestamps();

    // Get gps path
    nav_msgs::Path gps_path = ros_helpers::pointVectorToPath(gnss_points, "camera_init", kf_timestamps);
    this->publisher_gps_path_.publish(gps_path);

    // Make marker array
    std::vector<geometry_msgs::Point> gpsPts, kfPts;
    for (size_t i = 0; i < gnss_points.size(); i++) {
        if (!gnss_points.at(i).has_value()) continue;
        if(i >= updated_kf_poses.size()) continue;
        
        gpsPts.push_back(ros_helpers::vectorToPose(gnss_points.at(i).value()).position);
        kfPts.push_back(ros_helpers::isometryToPose(updated_kf_poses.at(i)).position);
    }

    auto blue = Eigen::Vector4d(0.0, 0.0, 1.0, 0.2);
    if (!gpsPts.empty() && (gpsPts.size() == kfPts.size())) {
        visualization_msgs::MarkerArray marker_arr = this->createMarkerArray(gpsPts, kfPts, blue, "camera_init", 0.05);
        this->publisher_gps_markers_.publish(marker_arr);
    }
    
}

/**
 * @brief Publishes the overall path, odometry, and broadcasts transform.
 */
void Visualizer::publishPath()
{
    if(!this->graph_manager_->graphInitialized() || this->graph_manager_->updatedKFIndex()<1) return;

    // Publish path from updated kf_poses
    auto updated_kf_poses = this->graph_manager_->getUpdatedKFPoses();
    updated_kf_poses.push_back(this->graph_manager_->updatedPose(true)); // Add the current pose estimate
    auto kf_timestamps    = this->graph_manager_->getKFTimestamps();
    nav_msgs::Path pgo_path = ros_helpers::isometryVectorToPath(updated_kf_poses, "camera_init", kf_timestamps);
    this->publisher_path_.publish(pgo_path);

    // Publish odometry from updated kf poses
    nav_msgs::Odometry pgo_odometry = ros_helpers::isometryToOdometry(
        graph_manager_->updatedPose(true),
        "camera_init", "/aft_pgo", ros::Time::now().toSec());
    this->publisher_odometry_.publish(pgo_odometry);

    // Update tranform
    tf::Transform transform = ros_helpers::isometryToTf(this->graph_manager_->updatedPose(true));
    tf_broadcaster_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "camera_init", "/aft_pgo"));
}

/**
 * @brief Publishes the global map, optimized from the pose graph.
 */
void Visualizer::publishMap(int frame_skip)
{
    // Assemble the global point cloud using updated poses
    pcl::PointCloud<PointType>::Ptr globalCloud = this->graph_manager_->assembleGlobalPointCloud(frame_skip);

    // Downsample the map into a separate cloud: filtering in place aliases input and output.
    pcl::PointCloud<PointType>::Ptr mapCloud(new pcl::PointCloud<PointType>());
    this->voxelizer_map_.setInputCloud(globalCloud);
    this->voxelizer_map_.filter(*mapCloud);

    // Convert the point cloud to ROS message and publish
    sensor_msgs::PointCloud2 mapMsg;
    pcl::toROSMsg(*mapCloud, mapMsg);
    mapMsg.header.frame_id = "camera_init";
    this->publisher_map_.publish(mapMsg);
}

/**
 * @brief Publishes the sensor cloud.
 */
void Visualizer::publishSensorCloud(const sensor_msgs::PointCloud2ConstPtr& ros_cloud)
{
    if (ros_cloud == nullptr) return;

    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr cloud_transformed(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*ros_cloud, *cloud);

    // Transform from lidar frame to world frame: T_world_lidar = T_world_body * T_body_lidar
    Eigen::Matrix4d T_world_lidar = this->graph_manager_->updatedPose(true).matrix() * this->params->extrinsics.T_body_lidar.matrix();
    * cloud_transformed = * geometry::pclTransform(cloud, T_world_lidar);

    // Convert the point cloud to ROS message and publish
    sensor_msgs::PointCloud2 sensorCloudMsg;
    pcl::toROSMsg(*cloud_transformed, sensorCloudMsg);
    sensorCloudMsg.header.frame_id = "camera_init";
    this->publisher_sensor_cloud_.publish(sensorCloudMsg);
}

/**
 * @brief Publishes the loop closure clouds and saves to a file, if desired
 * @param cloud_curr The current point cloud.
 * @param cloud_prev The previous point cloud.
 * @param kf_curr The index of the current keyframe.
 * @param kf_prev The index of the previous keyframe.
 */
void Visualizer::publishLCClouds(
    const pcl::PointCloud<PointType>::Ptr& cloud_curr,
    const pcl::PointCloud<PointType>::Ptr& cloud_prev,
    int kf_curr, int kf_prev)
{
    // Publishes the point clouds, if applicable
    if(this->params->icp.publish_pointclouds){
        sensor_msgs::PointCloud2 cloud_curr_msg;   
        pcl::toROSMsg(*cloud_curr, cloud_curr_msg);
        cloud_curr_msg.header.frame_id = "camera_init";     
        this->publisher_lc_cloud_curr_.publish(cloud_curr_msg);

        sensor_msgs::PointCloud2 cloud_prev_msg;
        pcl::toROSMsg(*cloud_prev, cloud_prev_msg);
        cloud_prev_msg.header.frame_id = "camera_init";
        this->publisher_lc_cloud_prev_.publish(cloud_prev_msg);
    }

    // Save the cloud to file, if applicable
    if(this->params->icp.save_pointclouds){
        std::stringstream filename_curr, filename_prev;
        filename_curr << this->params->ros.save_directory << "pcd_curr_" << kf_curr << "-" << kf_prev << ".pcd";
        filename_prev << this->params->ros.save_directory << "pcd_prev_" << kf_curr << "-" << kf_prev << ".pcd";
    
        pcl::io::savePCDFileASCII(filename_curr.str(), *cloud_curr);
        pcl::io::savePCDFileASCII(filename_prev.str(), *cloud_prev);
    }
}

void Visualizer::publishLCMarkers()
{
    if(!this->graph_manager_->graphInitialized() || this->graph_manager_->updatedKFIndex()<1) return;

    visualization_msgs::MarkerArray markerArray;

    auto kf_updated = this->graph_manager_->getUpdatedKFPoses();
    auto lc_manager = this->loop_closure_manager_.lock();
    if (!lc_manager){
        ROS_ERROR("Loop closure manager is not set. Cannot publish loop closure markers.");
        return;
    }
    std::vector<std::pair<int, int>> loop_closures = lc_manager->getAddedLoopClosures();

    if (loop_closures.empty()) return;

    std::vector<geometry_msgs::Point> lc_pnts_prev, lc_pnts_curr;

    // Transform isometries into points
    for (const auto& lc : loop_closures) {
        if(lc.first < 0 || lc.first >= static_cast<int>(kf_updated.size()) ||
           lc.second < 0 || lc.second >= static_cast<int>(kf_updated.size())) continue;

        lc_pnts_prev.push_back(ros_helpers::isometryToPose(kf_updated.at(lc.first)).position);
        lc_pnts_curr.push_back(ros_helpers::isometryToPose(kf_updated.at(lc.second)).position);
    }

    // publish
    auto red = Eigen::Vector4d(1.0, 0.0, 0.0, 0.4);
    visualization_msgs::MarkerArray marker_arr = this->createMarkerArray(lc_pnts_prev, lc_pnts_curr, red, "camera_init", 0.2);
    this->publisher_lc_markers_.publish(marker_arr);
}

/**
 * @brief Sets the loop closure manager.
 * @param lc_manager The loop closure manager to set.
 */
void Visualizer::setLoopClosureManager(std::shared_ptr<LoopClosureManager> lc_manager)
{
    this->loop_closure_manager_ = lc_manager;
}

/**
 * @brief Creates a marker array of marker connecting each pair of points from ptsA and ptsB.
 */
visualization_msgs::MarkerArray Visualizer::createMarkerArray(
    const std::vector<geometry_msgs::Point>& ptsA,
    const std::vector<geometry_msgs::Point>& ptsB,
    const Eigen::Vector4d& color,
    const std::string& frame_id,
    double line_width)
{
    visualization_msgs::MarkerArray markerArray;
    std_msgs::ColorRGBA color_obj;
    color_obj.r = color(0); color_obj.g = color(1); color_obj.b = color(2); color_obj.a = color(3);
    
    size_t N = std::min(ptsA.size(), ptsB.size());
    for (size_t i = 0; i < N; i++) {
        // Create marker
        visualization_msgs::Marker marker;
        marker.pose.orientation.w = 1; // avoids warning in rviz
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();

        // Set properties
        marker.color = color_obj;
        marker.scale.x = line_width;
        marker.id = i;
        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.action = visualization_msgs::Marker::ADD;

        // Add points
        marker.points.push_back(ptsA[i]);
        marker.points.push_back(ptsB[i]);

        // Push into array
        markerArray.markers.push_back(marker);
    }
    return markerArray;
}

}  // namespace spade_pgo