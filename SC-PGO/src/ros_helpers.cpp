#include "spade_pgo/ros_helpers.hpp"
#include <Eigen/Geometry>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_datatypes.h>
#include <ros/ros.h>
#include <string>

namespace spade_pgo {
namespace ros_helpers {

/**
 * @brief Converts an Eigen::Vector3d to a geometry_msgs::Pose.
 *
 * The translation is taken from the vector, and orientation is set to identity.
 */
geometry_msgs::Pose vectorToPose(const Eigen::Vector3d& v)
{
    geometry_msgs::Pose pose;
    pose.position.x = v.x();
    pose.position.y = v.y();
    pose.position.z = v.z();

    // Set identity orientation
    pose.orientation.x = 0.0;
    pose.orientation.y = 0.0;
    pose.orientation.z = 0.0;
    pose.orientation.w = 1.0;
    return pose;
}

/**
 * @brief Converts an Eigen::Isometry3d to a geometry_msgs::Pose message.
 * 
 * @param T The input transformation as an Eigen::Isometry3d.
 * @return geometry_msgs::Pose The resulting Pose message.
 */
geometry_msgs::Pose isometryToPose(const Eigen::Isometry3d& T)
{
    geometry_msgs::Pose pose;
    pose.position.x = T.translation().x();
    pose.position.y = T.translation().y();
    pose.position.z = T.translation().z();

    // Set identity orientation
    Eigen::Quaterniond q(T.rotation());
    pose.orientation.x = q.x();
    pose.orientation.y = q.y();
    pose.orientation.z = q.z();
    pose.orientation.w = q.w();

    return pose;
}

/**
 * @brief Converts an Eigen::Isometry3d to a nav_msgs::Odometry message.
 * 
 * @param T The input transformation as an Eigen::Isometry3d.
 * @param frame_id The frame ID for the output message.
 * @param child_frame_id The child frame ID for the output message.
 * @param stamp The timestamp for the output message.
 * @return nav_msgs::Odometry The resulting odometry message.
 */
nav_msgs::Odometry isometryToOdometry(const Eigen::Isometry3d& T, 
                                      const std::string& frame_id, 
                                      const std::string& child_frame_id,
                                      double stamp)
{
    ros::Time ros_stamp(stamp);
    nav_msgs::Odometry odom;
    odom.header.frame_id = frame_id;
    odom.child_frame_id = child_frame_id;
    odom.header.stamp = ros_stamp;
    odom.pose.pose = ros_helpers::isometryToPose(T);    
    return odom;
}

/**
 * @brief Converts an Eigen::Isometry3d to a tf::Transform.
 * 
 * @param T The input transformation as an Eigen::Isometry3d.
 * @return tf::Transform The resulting tf::Transform.
 */
tf::Transform isometryToTf(const Eigen::Isometry3d& T)
{
    // Create a tf::Transform and set its origin and rotation.
    tf::Vector3 origin(T.translation().x(), T.translation().y(), T.translation().z());
    Eigen::Quaterniond q(T.rotation());
    tf::Quaternion tf_q(q.x(), q.y(), q.z(), q.w());
    
    tf::Transform tfT;
    tfT.setOrigin(origin);
    tfT.setRotation(tf_q);
    
    return tfT;
}

/**
 * @brief Converts an Eigen::Isometry3d to a geometry_msgs::PoseStamped message.
 * 
 * Reuses the conversion logic from isometryToOdometry for the pose.
 * 
 * @param T The input transformation as an Eigen::Isometry3d.
 * @param frame_id The frame ID for the output message.
 * @param stamp The timestamp for the output message.
 * @return geometry_msgs::PoseStamped The resulting PoseStamped message.
 */
geometry_msgs::PoseStamped isometryToPoseStamped(const Eigen::Isometry3d& T, 
                                                   const std::string& frame_id,
                                                   double stamp)
{
    ros::Time ros_stamp(stamp);
    geometry_msgs::PoseStamped poseStamped;
    poseStamped.header.frame_id = frame_id;
    poseStamped.header.stamp = ros_stamp;
    poseStamped.pose = isometryToPose(T);
    
    return poseStamped;
}

/**
 * @brief Converts a vector of Eigen::Isometry3d to a nav_msgs::Path message.
 * 
 * @param poses The input vector of transformations.
 * @param frame_id The frame ID for the output message.
 * @param timestamps The timestamps for each pose in the vector.
 * @return nav_msgs::Path The resulting path message.
 */
nav_msgs::Path isometryVectorToPath(
    const std::vector<Eigen::Isometry3d>& poses, 
    const std::string& frame_id,
    const std::vector<double> timestamps )
{
    nav_msgs::Path path;
    path.header.frame_id = frame_id;

    for (size_t i = 0; i < poses.size(); ++i) {
        double ts = (!timestamps.empty() && i < timestamps.size()) ? timestamps[i] : ros::Time::now().toSec();
        geometry_msgs::PoseStamped poseStamped = isometryToPoseStamped(poses[i], frame_id, ts);
        path.poses.push_back(poseStamped);
    }

    return path;
}

/**
 * @brief Converts a vector of Eigen::Vector3d to a nav_msgs::Path message.
 * 
 * @param points The input vector of points.
 * @param frame_id The frame ID for the output message.
 * @param timestamps The timestamps for each pose in the vector.
 * @return nav_msgs::Path The resulting path message.
 */
nav_msgs::Path pointVectorToPath(
    const std::vector<Eigen::Vector3d>& points, 
    const std::string& frame_id,
    const std::vector<double> timestamps )
{
    // Convert points to an isometry
    std::vector<Eigen::Isometry3d> isometry_points;
    isometry_points.reserve(points.size());
    for (const auto& point:points){
        Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
        iso.translation() = point;
        isometry_points.push_back(iso);
    }
    return isometryVectorToPath(isometry_points, frame_id, timestamps );
}

/**
 * @brief Converts a vector of std::optional<Eigen::Vector3d> to a nav_msgs::Path message.
 * 
 * @param points The input vector of points.
 * @param frame_id The frame ID for the output message.
 * @param timestamps The timestamps for each pose in the vector.
 * @return nav_msgs::Path The resulting path message.
 */
nav_msgs::Path pointVectorToPath(
    const std::vector<std::optional<Eigen::Vector3d>>& points, 
    const std::string& frame_id,
    const std::vector<double> timestamps )
{
    // Convert points to an isometry
    std::vector<Eigen::Isometry3d> isometry_points;
    isometry_points.reserve(points.size());
    for (const auto& point:points){
        if (!point.has_value()) continue;
        Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
        iso.translation() = point.value();
        isometry_points.push_back(iso);
    }
    return isometryVectorToPath(isometry_points, frame_id, timestamps );
}

} // namespace ros_helpers
} // namespace spade_pgo