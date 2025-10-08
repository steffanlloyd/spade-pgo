// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

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

inline void saveTransformsKITTI(std::vector<Eigen::Isometry3d> transforms, std::string filename)
{
  std::fstream stream(filename.c_str(), std::fstream::out);
  for(const auto& Ti: transforms) {
    stream << Ti(0,0) << " " << Ti(0,1) << " " << Ti(0,2) << " " << Ti(0,3) << " "
           << Ti(1,0) << " " << Ti(1,1) << " " << Ti(1,2) << " " << Ti(1,3) << " "
           << Ti(2,0) << " " << Ti(2,1) << " " << Ti(2,2) << " " << Ti(2,3) << std::endl;
  }

}

}