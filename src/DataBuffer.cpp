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
    
    // Extract the odometry timestamp.
    double timestamp_lio = this->odom_buffer_.front()->header.stamp.toSec();
    double timestamp_laser = this->cloud_buffer_.front()->header.stamp.toSec();

    // Find a nearby GNSS message
    std::optional<sensor_msgs::NavSatFix::ConstPtr> gnss_msg;
    while (!this->gnss_buffer_.empty()) {
        auto msg = this->gnss_buffer_.front();
        this->gnss_buffer_.pop();
        if( abs(msg->header.stamp.toSec() - timestamp_lio) < this->gnssAllowedTimeDelta ) {
            gnss_msg = msg;
            break;
        }
    }
    
    // Convert the full-resolution cloud message to a PCL point cloud.
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*this->cloud_buffer_.front(), *cloud);
    this->cloud_buffer_.pop();
    
    // Convert the odometry message to an Isometry3d.
    Eigen::Isometry3d T_odom = this->odomMsgToIsometry(*this->odom_buffer_.front());
    this->odom_buffer_.pop();
    
    return std::make_optional(DataPoint{timestamp_lio, timestamp_laser, cloud, T_odom, gnss_msg});
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
