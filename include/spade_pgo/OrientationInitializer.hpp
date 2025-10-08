#pragma once

#include <deque>
#include <mutex>
#include <optional>
#include <string>

// ROS includes
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float64.h>

// PGO includes
#include "spade_pgo/common.hpp"

// Eigen includes
#include <Eigen/Geometry>
#include <Eigen/Dense>

using namespace spade_pgo;

namespace spade_pgo {

class OrientationInitializer{
public:
    OrientationInitializer( int calibration_size = 10 )
        : calibration_size_(calibration_size){}

    void pushIMU(const sensor_msgs::Imu::ConstPtr& msg);
    void pushHeading(const std_msgs::Float64::ConstPtr& msg);
    bool bufferReady();

    std::optional<Eigen::Matrix3d> getOrientation();

private:
    size_t calibration_size_ = 10;

    std::deque<Eigen::Vector3d> accel_buffer_;
    std::deque<double> heading_buffer_;
};

} // End of namespace spade_pgo