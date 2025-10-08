#include "spade_pgo/common.hpp"
#include "spade_pgo/geometry.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <omp.h>

#include <small_gicp/points/point_cloud.hpp>

using namespace spade_pgo;

/**
 * @brief Computes the linear and angular distance of a transform.
 *
 * @param Tdelta The affine transformation.
 * @return A pair containing the linear and angular distance.
 */
std::pair<double, double> spade_pgo::geometry::poseDistance(const Eigen::Affine3d& Tdelta)
{
    double linear_distance = Tdelta.translation().norm();
    Eigen::AngleAxisd angle_axis(Tdelta.rotation());
    double angular_distance = std::abs(angle_axis.angle());

    return std::pair<double, double>(linear_distance, angular_distance);
}

/**
 * @brief Computes the linear and angular distance between two poses.
 *
 * @param T1 The first affine transformation.
 * @param T2 The second affine transformation.
 * @return A pair containing the linear and angular distance.
 */
std::pair<double, double> spade_pgo::geometry::poseDistance(const Eigen::Affine3d& T1, const Eigen::Affine3d& T2 )
{
    return geometry::poseDistance(T1.inverse() * T2);
}

/**
 * @brief Transforms a point cloud using a given transformation matrix.
 *
 * @param pointcloud The input point cloud.
 * @param T The transformation matrix.
 * @return The transformed point cloud.
 */
pcl::PointCloud<PointType>::Ptr spade_pgo::geometry::pclTransform(const pcl::PointCloud<PointType>::Ptr &pointcloud, const Eigen::Matrix4d T)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = pointcloud->size();
    cloudOut->resize(cloudSize);
    Eigen::Matrix4f T_float = T.cast<float>();
    
    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = pointcloud->points[i];
        cloudOut->points[i].x = T_float(0,0) * pointFrom.x + T_float(0,1) * pointFrom.y + T_float(0,2) * pointFrom.z + T_float(0,3);
        cloudOut->points[i].y = T_float(1,0) * pointFrom.x + T_float(1,1) * pointFrom.y + T_float(1,2) * pointFrom.z + T_float(1,3);
        cloudOut->points[i].z = T_float(2,0) * pointFrom.x + T_float(2,1) * pointFrom.y + T_float(2,2) * pointFrom.z + T_float(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}

/**
 * @brief Computes the squared Euclidean distance squared between two poses.
 *
 * @param T1 The first pose.
 * @param T2 The second pose.
 * @return The squared Euclidean distance.
 */
double spade_pgo::geometry::euclideanDistance2(const Eigen::Isometry3d& T1, const Eigen::Isometry3d& T2) {
    return (T1.translation() - T2.translation()).squaredNorm();
}

/**
 * @brief Computes the Euclidean distance between two poses.
 *
 * @param T1 The first pose.
 * @param T2 The second pose.
 * @return The Euclidean distance.
 */
double spade_pgo::geometry::euclideanDistance(const Eigen::Isometry3d& T1, const Eigen::Isometry3d& T2) {
    return (T1.translation() - T2.translation()).norm();
}

/**
 * Converts a pcl point cloud to an Eigen point cloud.
 * @param pcl_cloud The input pcl point cloud.
 * @return The converted Eigen point cloud.
 */
std::shared_ptr<small_gicp::PointCloud> spade_pgo::geometry::pclToEigen(const pcl::PointCloud<PointType>::Ptr& pcl_cloud) {
    std::vector<Eigen::Vector3d> points;
    points.reserve(pcl_cloud->points.size());  // Reserve space for efficiency

    for (const auto& point : pcl_cloud->points) {
        points.emplace_back(point.x, point.y, point.z);
    }

    return std::make_shared<small_gicp::PointCloud>(points);
}

/**
 * Converts an Eigen point cloud to a pcl point cloud.
 * @param eigen_cloud The input Eigen point cloud.
 * @return The converted pcl point cloud.
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr spade_pgo::geometry::eigenToPcl(const std::shared_ptr<small_gicp::PointCloud> eigen_cloud) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl_cloud->resize(eigen_cloud->size());
    for (size_t i = 0; i<eigen_cloud->size(); ++i)
    {
        pcl_cloud->points[i].x = eigen_cloud->points[i][0];
        pcl_cloud->points[i].y = eigen_cloud->points[i][1];
        pcl_cloud->points[i].z = eigen_cloud->points[i][2];
    }
    return pcl_cloud;
}