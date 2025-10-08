#include <deque>
#include <mutex>
#include <optional>
#include <string>

// PGO includes
#include "spade_pgo/common.hpp"
#include "spade_pgo/OrientationInitializer.hpp"

// ROS includes
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float64.h>

// Eigen includes
#include <Eigen/Geometry>
#include <Eigen/Dense>

using namespace spade_pgo;

/**
 * @brief Pushes an IMU message into the acceleration buffer.
 *
 * @param msg A constant pointer to the incoming sensor_msgs::Imu message.
 */
void OrientationInitializer::pushIMU(const sensor_msgs::Imu::ConstPtr& msg)
{
    Eigen::Vector3d accel(  msg->linear_acceleration.x,
        msg->linear_acceleration.y,
        msg->linear_acceleration.z);

    this->accel_buffer_.push_back(accel);
    if (this->accel_buffer_.size() > this->calibration_size_)
        this->accel_buffer_.pop_front();
}

/**
 * @brief Pushes a heading message into the heading buffer.
 *
 * @param msg A std_msgs::Float64 message containing the heading in degrees.
 */
void OrientationInitializer::pushHeading(const std_msgs::Float64::ConstPtr& msg)
{
    this->heading_buffer_.push_back(msg->data * M_PI / 180.0);
    if (this->heading_buffer_.size() > this->calibration_size_)
        this->heading_buffer_.pop_front();
}

/**
 * @brief Checks if the buffers are ready for calibration.
 * @return true if both buffers are full
 */
bool OrientationInitializer::bufferReady()
{
    return  this->accel_buffer_.size() == this->calibration_size_ &&
            this->heading_buffer_.size() == this->calibration_size_;
}

/**
 * @brief Computes the initial orientation based on buffered IMU and heading data.
 *
 * If the acceleration and heading buffers are full, the function computes the average heading
 * and acceleration to derive an orientation. The "up" vector is obtained by normalizing the averaged
 * acceleration, and a desired horizontal y-axis is constructed from the average heading.
 * The horizontal y-axis is then orthogonalized with respect to the "up" vector, and the x-axis is 
 * computed as the cross product of this orthogonalized y-axis and the up vector. These axes are used
 * to form a rotation matrix.
 *
 * @return std::optional<Matrix3d> A rotation matrix representing the initial orientation if the
 *         buffers are ready; otherwise, returns std::nullopt and logs a warning.
 */
std::optional<Eigen::Matrix3d> OrientationInitializer::getOrientation()
{
    if(!this->bufferReady()){
        return std::nullopt;
    }

    // Average the heading buffer
    double sum_heading = 0;
    for (const auto& heading : this->heading_buffer_) sum_heading += heading;
    double avg_heading = sum_heading / this->heading_buffer_.size();

    // Average the acceleration buffer
    Eigen::Vector3d sum_accel = Eigen::Vector3d::Zero();
    for (const auto& accel : this->accel_buffer_) sum_accel += accel;
    Eigen::Vector3d avg_accel = sum_accel / this->accel_buffer_.size();

    // Normalize the average acceleration to obtain the "up" vector.
    Eigen::Vector3d up = avg_accel.normalized();

    // Construct the desired horizontal y-axis from the average heading.
    // For a heading of 0, assume true north corresponds to (0, 1, 0).
    Eigen::Vector3d y_desired(std::sin(avg_heading), std::cos(avg_heading), 0.0);
    y_desired.normalize();

    // Project y_desired onto the horizontal plane orthogonal to up.
    Eigen::Vector3d y_aligned = y_desired - (y_desired.dot(up)) * up;
    y_aligned.normalize();

    // Compute the x-axis as the cross product of y_aligned and up.
    Eigen::Vector3d x_axis = y_aligned.cross(up).normalized();

    // Build a rotation matrix with the computed orthonormal basis.
    Eigen::Matrix3d R;
    // Columns: x_axis, y_aligned, up (z-axis)
    R.col(0) = x_axis;
    R.col(1) = y_aligned;
    R.col(2) = up;

    return R;
}