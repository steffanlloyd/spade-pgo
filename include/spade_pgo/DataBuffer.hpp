#pragma once

#include <queue>
#include <mutex>
#include <optional>
#include <tuple>
#include <string>

// PGO includes
#include "spade_pgo/common.hpp"

// ROS includes
#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Quaternion.h>

// PCL includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>

// Eigen includes
#include <Eigen/Geometry>
#include <Eigen/Dense>

using namespace spade_pgo;

namespace spade_pgo {

class DataBuffer {
public:
    DataBuffer(double gnssAllowedTimeDelta = 0.1)
        : gnssAllowedTimeDelta(gnssAllowedTimeDelta){}

    void pushOdometry(const nav_msgs::Odometry::ConstPtr& odom);
    void pushPointCloud(const sensor_msgs::PointCloud2ConstPtr& cloud);
    void pushGNSS(const sensor_msgs::NavSatFix::ConstPtr &gps);

    void pushOrientation(const nav_msgs::Odometry::ConstPtr& odom);
    void pushOrientation(const geometry_msgs::Quaternion::ConstPtr& quat);    

    std::optional<DataPoint> popDataPoint();
    bool dataAvailable(bool lock_mutex = true);
    void clearAll();

    double gnssAllowedTimeDelta;

private:
    std::queue<nav_msgs::Odometry::ConstPtr> odom_buffer_;
    std::queue<sensor_msgs::PointCloud2ConstPtr> cloud_buffer_;
    std::queue<sensor_msgs::NavSatFix::ConstPtr> gnss_buffer_;
    std::queue<Eigen::Quaterniond> orientation_buffer_;
    mutable std::mutex buffer_mutex_;
    
    /// Helper function to convert a nav_msgs::Odometry message into an Eigen::Isometry3d.
    Eigen::Isometry3d odomMsgToIsometry(const nav_msgs::Odometry &odom);
};

}