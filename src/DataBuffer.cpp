#include "spade_pgo/DataBuffer.hpp"

#include "spade_pgo/common.hpp"

// Standard includes
#include <optional>
#include <tuple>
#include <queue>
#include <mutex>
#include <string>

// Eigen includes
#include <Eigen/Geometry>
#include <Eigen/Dense>

// ROS includes
#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>

// PCL conversions
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace spade_pgo;

namespace spade_pgo {

    
/**
 * @brief Pushes an odometry message into the odometry buffer.
 * 
 * @param odom A constant pointer to the odometry message to be pushed into the buffer.
 */
void DataBuffer::pushOdometry(const nav_msgs::Odometry::ConstPtr& odom)
{
    std::lock_guard<std::mutex> lock(this->buffer_mutex_);
    this->odom_buffer_.push(odom);
}

/**
 * @brief Pushes a point cloud message into the point cloud buffer.
 * 
 * @param cloud A constant pointer to the point cloud message to be pushed into the buffer.
 */
void DataBuffer::pushPointCloud(const sensor_msgs::PointCloud2ConstPtr& cloud)
{
    std::lock_guard<std::mutex> lock(this->buffer_mutex_);
    this->cloud_buffer_.push(cloud);
}

/**
 * @brief Pushes a GNSS message into the GNSS buffer.
 * 
 * @param cloud A constant pointer to the point cloud message to be pushed into the buffer.
 */
void DataBuffer::pushGNSS(const sensor_msgs::NavSatFix::ConstPtr &gps)
{
    std::lock_guard<std::mutex> lock(this->buffer_mutex_);
    this->gnss_buffer_.push(gps);
}

/**
 * @brief Pushes an orientation message into the orientation buffer (from an odometry message).
 * 
 * @param odom A constant pointer to the odometry message to be pushed into the buffer.
 */
void DataBuffer::pushOrientation(const nav_msgs::Odometry::ConstPtr& odom)
{
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    Eigen::Quaterniond q(
        odom->pose.pose.orientation.w,
        odom->pose.pose.orientation.x,
        odom->pose.pose.orientation.y,
        odom->pose.pose.orientation.z
    );
    this->orientation_buffer_.push(q.normalized());
}


/**
 * @brief Pushes an orientation message into the orientation buffer (from a quaternion message).
 * 
 * @param quat A constant pointer to the quaternion message to be pushed into the buffer.
 */
void DataBuffer::pushOrientation(const geometry_msgs::Quaternion::ConstPtr& quat)
{
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    Eigen::Quaterniond q(quat->w, quat->x, quat->y, quat->z);
    this->orientation_buffer_.push(q.normalized());
}



/**
 * @brief Pops synchronized odometry and point cloud data from the buffers.
 * 
 * This function attempts to pop synchronized odometry and
 * point cloud data from their respective buffers. If either buffer is empty, it returns
 * an empty optional. Otherwise, it synchronizes the buffers by discarding older odometry
 * messages, converts the point cloud message to a PCL point cloud, and converts the
 * odometry message to an Isometry3d transformation.
 * It also gets a matching GPS message, if available.
 * 
 * @return An optional DataPoint struct containing the timestamp, point cloud, and odometry transformation.  If synchronization fails due to empty buffers, returns std::nullopt.
 */
std::optional<DataPoint> DataBuffer::popDataPoint()
{
    std::lock_guard<std::mutex> lock(this->buffer_mutex_);
        
    // Synchronize buffers: discard odometry messages older than the cloud.
    while (this->dataAvailable() &&
           this->odom_buffer_.front()->header.stamp.toSec() < this->cloud_buffer_.front()->header.stamp.toSec())
    {
        this->odom_buffer_.pop();
    }
    
    // After synchronization, verify buffers are still non-empty.
    if (! this->dataAvailable()) return std::nullopt;

    DataPoint data;
    
    // Extract the odometry timestamp.
    data.timestamp_lio = this->odom_buffer_.front()->header.stamp.toSec();
    data.timestamp_laser = this->cloud_buffer_.front()->header.stamp.toSec();

    // Find a nearby GNSS message
    while (!this->gnss_buffer_.empty()) {
        auto msg = this->gnss_buffer_.front();
        this->gnss_buffer_.pop();
        if( abs(msg->header.stamp.toSec() - data.timestamp_lio) < this->gnssAllowedTimeDelta ) {
            data.gnss_msg = msg;
            break;
        }
    }

    // Get latest orientation if available
    if (!this->orientation_buffer_.empty()) {
        data.orientation_msg = this->orientation_buffer_.back();
        // Clear old orientations, keep only the most recent
        while (orientation_buffer_.size() > 1) {
            orientation_buffer_.pop();
        }
    }
    
    // Convert the full-resolution cloud message to a PCL point cloud.
    data.pointcloud = boost::make_shared<pcl::PointCloud<PointType>>();
    pcl::fromROSMsg(*this->cloud_buffer_.front(), *data.pointcloud);
    this->cloud_buffer_.pop();
    
    // Convert the odometry message to an Isometry3d.
    data.T_odom = this->odomMsgToIsometry(*this->odom_buffer_.front());
    this->odom_buffer_.pop();
    
    return std::move(data);
}

/**
 * Checks whether there is data in the buffers
 */
bool DataBuffer::dataAvailable()
{
    return !this->odom_buffer_.empty() && !this->cloud_buffer_.empty();
}


/**
 * @brief Converts an odometry message to an Isometry3d transformation.
 * 
 * @param nav_msgs::Odometry odom The odometry message to be converted.
 * @return Isometry3d The resulting Isometry3d transformation.
 */
Eigen::Isometry3d DataBuffer::odomMsgToIsometry(const nav_msgs::Odometry &odom)
{
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    // Set the translation part
    T.translation() << odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z;
    
    // Convert quaternion to rotation matrix
    Eigen::Quaterniond q(
        odom.pose.pose.orientation.w,
        odom.pose.pose.orientation.x,
        odom.pose.pose.orientation.y,
        odom.pose.pose.orientation.z
    );
    T.linear() = q.toRotationMatrix();
    
    return T;
}

} // namespace spade_pgo
