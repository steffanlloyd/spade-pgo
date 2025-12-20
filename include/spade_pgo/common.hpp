#pragma once

#include <cmath>
#include <pcl/point_types.h>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sensor_msgs/NavSatFix.h>

namespace spade_pgo{

typedef pcl::PointXYZI PointType;

inline double rad2deg(double radians)
{
	return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees)
{
	return degrees * M_PI / 180.0;
}

struct Pose6D {
	double x;
	double y;
	double z;
	double roll;
	double pitch;
	double yaw;
};

struct DataPoint {
	double timestamp_lio;
	double timestamp_laser;
	pcl::PointCloud<PointType>::Ptr pointcloud;
	Eigen::Isometry3d T_odom;
	std::optional<sensor_msgs::NavSatFix::ConstPtr> gnss_msg;
	std::optional<Eigen::Quaterniond> orientation_msg; 
};

struct LoopClosure {
	int id1;
	int id2;
	Eigen::Isometry3d T;
};

inline std::string isometryToStr(Eigen::Isometry3d T){
	// Extract translation
	Eigen::Vector3d translation = T.translation();
	Eigen::Vector3d rpy = T.rotation().eulerAngles(2,1,0);
	
	// ROS_INFO("Translation: x = %f, y = %f, z = %f, roll = %f, pitch = %f, yaw = %f", translation.x(), translation.y(), translation.z(), roll, pitch, yaw);
	// Format translation and rotation as a string
	char buffer[200];
	std::snprintf(buffer, sizeof(buffer), "{x = %.4f, y = %.4f, z = %.4f, roll = %.4f, pitch = %.4f, yaw = %.4f}", translation.x(), translation.y(), translation.z(), rpy(0), rpy(1), rpy(2));
	
	return std::string(buffer);
}

inline std::string padZeros(int val, int num_digits = 6) {
	std::ostringstream out;
	out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
	return out.str();
}

inline void saveTransformsToFile(
    const std::vector<Eigen::Isometry3d>& transforms,
    const std::vector<uint8_t>& drone_ids,
    const std::string& filename)
{
    std::fstream stream(filename.c_str(), std::fstream::out);

    // Header
    stream << "# keyframe_id drone_id r00 r01 r02 t0 r10 r11 r12 t1 r20 r21 r22 t2" << std::endl;

    for (size_t i = 0; i < transforms.size(); ++i) {
        const auto& Ti = transforms[i];
        uint8_t drone_id = (i < drone_ids.size()) ? drone_ids[i] : 0;

        stream << i << " " << static_cast<int>(drone_id) << " "
               << Ti(0,0) << " " << Ti(0,1) << " " << Ti(0,2) << " " << Ti(0,3) << " "
               << Ti(1,0) << " " << Ti(1,1) << " " << Ti(1,2) << " " << Ti(1,3) << " "
               << Ti(2,0) << " " << Ti(2,1) << " " << Ti(2,2) << " " << Ti(2,3) << std::endl;
    }
}

} // namespace spade_pgo