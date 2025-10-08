#pragma once

#include "spade_pgo/common.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <small_gicp/points/point_cloud.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace spade_pgo;

namespace spade_pgo{
namespace geometry{

std::pair<double, double> poseDistance(const Eigen::Affine3d& Tdelta);
std::pair<double, double> poseDistance(const Eigen::Affine3d& T1, const Eigen::Affine3d& T2 );

pcl::PointCloud<PointType>::Ptr pclTransform(const pcl::PointCloud<PointType>::Ptr &pointcloud, const Eigen::Matrix4d T);

double euclideanDistance2(const Eigen::Isometry3d& T1, const Eigen::Isometry3d& T2);
double euclideanDistance(const Eigen::Isometry3d& T1, const Eigen::Isometry3d& T2);

std::shared_ptr<small_gicp::PointCloud> pclToEigen(const pcl::PointCloud<PointType>::Ptr& pcl_cloud);
pcl::PointCloud<pcl::PointXYZ>::Ptr eigenToPcl(const std::shared_ptr<small_gicp::PointCloud> eigen_cloud);

} // End namespace geometry
} // End namespace spade_pgo