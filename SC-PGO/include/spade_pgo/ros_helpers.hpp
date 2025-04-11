#pragma once

#include <Eigen/Geometry>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_datatypes.h>
#include <ros/ros.h>
#include <string>
#include <vector>

namespace spade_pgo {
namespace ros_helpers {

geometry_msgs::Pose vectorToPose(const Eigen::Vector3d& v);
geometry_msgs::Pose isometryToPose(const Eigen::Isometry3d& T);

nav_msgs::Odometry isometryToOdometry(
    const Eigen::Isometry3d& T, 
    const std::string& frame_id = "map", 
    const std::string& child_frame_id = "base_link",
    double stamp = ros::Time::now().toSec());

tf::Transform isometryToTf(const Eigen::Isometry3d& T);

geometry_msgs::PoseStamped isometryToPoseStamped(
    const Eigen::Isometry3d& T, 
    const std::string& frame_id = "map",
    double stamp = ros::Time::now().toSec());

nav_msgs::Path isometryVectorToPath(
    const std::vector<Eigen::Isometry3d>& poses, 
    const std::string& frame_id = "map",
    const std::vector<double> timestamps = std::vector<double>());

nav_msgs::Path pointVectorToPath(
    const std::vector<Eigen::Vector3d>& poses, 
    const std::string& frame_id = "map",
    const std::vector<double> timestamps = std::vector<double>());
    
nav_msgs::Path pointVectorToPath(
    const std::vector<std::optional<Eigen::Vector3d>>& poses, 
    const std::string& frame_id = "map",
    const std::vector<double> timestamps = std::vector<double>());


} // namespace ros_helpers
} // namespace spade_pgo