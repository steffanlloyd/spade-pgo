#pragma once

#include <mutex>
#include <queue>
#include <vector>
#include <optional>

#include <Eigen/Geometry>
#include <Eigen/Dense>

// SPADE PGO includes
#include "spade_pgo/common.hpp"
#include "spade_pgo/PGOParams.hpp"
#include "spade_pgo/DataBuffer.hpp"
#include "spade_pgo/OrientationInitializer.hpp"

// ROS includes
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

// GTSAM includes
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Pose3.h>

#include <GeographicLib/LocalCartesian.hpp>


using namespace spade_pgo;

namespace spade_pgo{
// Forward declaration
class LoopClosureManager;

class PoseGraphManager {
public:
    PoseGraphManager(std::shared_ptr<PGOParams> params);

    bool graphInitialized() const;
    int updatedKFIndex() const;
    Eigen::Isometry3d updatedPose(bool live=false) const;
    int currentKFIndex() const;
    int graphSize() const;
    std::vector<Eigen::Isometry3d> getUpdatedKFPoses() const;
    std::vector<std::optional<Eigen::Vector3d>> getGNSSPoints() const;
    std::vector<double> getKFTimestamps() const;
    void triggerExtraOptimization();

    void setCurrentPose(const Eigen::Isometry3d &pose);
    void setLoopClosureManager(std::shared_ptr<LoopClosureManager> lc_manager);

    void processData(DataPoint data);
    int addKeyframe(const Eigen::Isometry3d &T, pcl::PointCloud<PointType>::Ptr &pointcloud, double time);
    void addPriorFactorandEstimate(const Eigen::Isometry3d &T);
    void addOdometryFactorAndEstimate(int kf_index, const Eigen::Isometry3d &T, Eigen::Isometry3d &T_est);
    void addGNSSFactor(int kf_index, nav_msgs::Odometry::ConstPtr gnss_odom);
    void addLoopClosureFactor(int kf_index_1, int kf_index_2, Eigen::Isometry3d Ticp, double fitnessScore);

    nav_msgs::Odometry::ConstPtr navSatFixToOdometry(const sensor_msgs::NavSatFix::ConstPtr& nav_sat_fix, bool resetOrigin = false);

    void optimizeGraph();
    void savePointCloud(int kf_index, pcl::PointCloud<PointType>::Ptr pointcloud, double timestamp) const;
    void saveGraphKeyframes() const;

    pcl::PointCloud<PointType>::Ptr assembleGlobalPointCloud(int frame_skip = 1) const;

    gtsam::NonlinearFactorGraph graph;
    DataBuffer data_buffer;
    std::shared_ptr<PGOParams> params;
    std::unique_ptr<OrientationInitializer> orienter;
    std::vector<pcl::PointCloud<PointType>::Ptr> kf_pointclouds; 

private:
    double distanceTravelledGNSS_() const;
    Eigen::Isometry3d estimateUpdatedPose(Eigen::Isometry3d T_odom) const;

    mutable std::mutex kf_mtx_;
    mutable std::mutex kf_updated_mtx_;
    mutable std::mutex graph_mtx_;
    mutable std::mutex current_pose_mtx_;
    std::weak_ptr<LoopClosureManager> loop_closure_manager_;

    gtsam::Values kf_isam_estimates_;
    gtsam::ISAM2 isam_;
    GeographicLib::LocalCartesian geo_converter_;

    Eigen::Isometry3d T0_ = Eigen::Isometry3d::Identity();
    std::vector<Eigen::Isometry3d> kf_poses_odom_;
    std::vector<Eigen::Isometry3d> kf_poses_updated_;
    std::vector<std::optional<Eigen::Vector3d>> kf_gnss_;
    std::vector<double> kf_timestamps_;
    std::queue<gtsam::GPSFactor> gnss_factor_buffer_;
    bool graph_initialized_ = false;
    bool gnss_initialized_ = false;
    std::optional<Eigen::Vector3d> gnss_origin_; // Will store values as lat, long, altitude
    bool trigger_extra_optimization_flag_ = false;
    Eigen::Isometry3d T_curr_updated_ = Eigen::Isometry3d::Identity();

    std::queue<std::pair<Eigen::Isometry3d, pcl::PointCloud<PointType>::Ptr>> inter_kf_pointcloud_buffer_;

    pcl::VoxelGrid<PointType> voxelizer_sc_;
    pcl::VoxelGrid<PointType> voxelizer_save_;

    mutable std::fstream pg_laser_timestamp_save_;
};

}