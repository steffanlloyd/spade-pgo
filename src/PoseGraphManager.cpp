#include "spade_pgo/PoseGraphManager.hpp"

#include <mutex>
#include <queue>
#include <vector>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>

// SPADE PGO includes
#include "spade_pgo/common.hpp"
#include "spade_pgo/geometry.hpp"
#include "spade_pgo/PGOParams.hpp"
#include "spade_pgo/LoopClosureManager.hpp"
#include "scancontext/Scancontext.h"

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

// GeographicLib
#include <GeographicLib/LocalCartesian.hpp>

#include "scancontext/Scancontext.h"

using namespace spade_pgo;

/**
 * @brief Constructs a PoseGraphManager object with the given parameters.
 * 
 * @param params Shared pointer to the parameters object containing configuration settings.
 */
PoseGraphManager::PoseGraphManager(
    std::shared_ptr<PGOParams> params
) : params(params)
{
    // Build isam optimizer
    gtsam::ISAM2Params isam_params;
    isam_params.relinearizeThreshold = 0.01;
    isam_params.relinearizeSkip = 1;
    this->isam_ = gtsam::ISAM2(isam_params);

    this->data_buffer.gnssAllowedTimeDelta = this->params->graph.gnss_time_delta;

    // Setup voxel filters
    this->voxelizer_sc_.setLeafSize(this->params->sc.voxel_size, this->params->sc.voxel_size, this->params->sc.voxel_size);
    double voxel_size_save = std::min(this->params->icp.voxel_size, this->params->sc.voxel_size);
    this->voxelizer_save_.setLeafSize(voxel_size_save, voxel_size_save, voxel_size_save);

    // Setup outputs
    this->pg_laser_timestamp_save_ = std::fstream(this->params->ros.save_directory + "laser_timestamps.txt", std::fstream::out);
    this->pg_laser_timestamp_save_.precision(std::numeric_limits<double>::max_digits10);
}

/**
 * @brief Adds a keyframe to the pose graph (consisting of, at a minimum, an transform estimate and pointcloud)
 * 
 * @param T The transformation matrix representing the pose of the keyframe.
 * @param pointcloud Pointer to the point cloud data associated with the keyframe.
 * @param time The timestamp of the keyframe.
 * @return The index of the added keyframe.
 */
int PoseGraphManager::addKeyframe(const Eigen::Isometry3d &T, pcl::PointCloud<PointType>::Ptr &pointcloud, double time)
{
    // Downsample for visualization and loop closure
    pcl::PointCloud<PointType>::Ptr pointcloud_downsampled(new pcl::PointCloud<PointType>());
    this->voxelizer_save_.setInputCloud(pointcloud);
    this->voxelizer_save_.filter(*pointcloud_downsampled);
    
    // push into queue
    int kf_id;
    {
        std::lock_guard<std::mutex> lock(this->kf_mtx_);
        this->kf_pointclouds.push_back(pointcloud_downsampled);
        this->kf_poses_odom_.push_back(T);
        this->kf_timestamps_.push_back(time);
        // Track which session this keyframe belongs to (for multi-session processing)
        this->kf_session_ids_.push_back(this->current_session_id_);
        kf_id = this->kf_poses_odom_.size() - 1;
    }

    ROS_DEBUG("Added keyframe %d, estimated position: %s", kf_id, isometryToStr(T).c_str());

    // Add to scancontext
    // TO DO: Fix this scancontrol stuff
    if(this->params->sc.enabled){
        // Downsample input point cloud from pointcloud according to voxel for SC
        pcl::PointCloud<PointType>::Ptr pointcloud_downsampled_sc(new pcl::PointCloud<PointType>());
        this->voxelizer_sc_.setInputCloud(pointcloud);
        this->voxelizer_sc_.filter(*pointcloud_downsampled_sc);

        auto loop_closure_manager = this->loop_closure_manager_.lock();
        if(loop_closure_manager) loop_closure_manager->sc_detector.makeAndSaveScancontextAndKeys(*pointcloud_downsampled_sc);
        else ROS_ERROR("Loop closure manager is not set. Cannot add scancontext.");
    }

    return kf_id;
}

/**
 * @brief Checks if the pose graph has been initialized.
 */
bool PoseGraphManager::graphInitialized() const
{
    std::lock_guard<std::mutex> lock(this->kf_mtx_);
    return this->kf_poses_odom_.size() > 0;
}

/**
 * @brief Gets the index of the most recent updated keyframe.
 */
int PoseGraphManager::updatedKFIndex() const
{
    std::lock_guard<std::mutex> lock(this->kf_updated_mtx_);
    return this->kf_poses_updated_.size() - 1;
}

/**
 * @brief Gets the index of the most recent updated keyframe.
 * @param live If true, returns the current updated pose. If false, returns the last updated keyframe pose.
 */
Eigen::Isometry3d PoseGraphManager::updatedPose(bool live) const
{
    if(live){
        std::lock_guard<std::mutex> lock(this->current_pose_mtx_);
        return this->T_curr_updated_;
    }else{
        std::lock_guard<std::mutex> lock(this->kf_updated_mtx_);

        // Handle case where graph optimization hasn't been run yet.
        if(this->kf_poses_updated_.empty()){
            if(this->graphInitialized()){
                return this->T0_;
            }else{
                ROS_ERROR("No updated poses available in updatePose.");
                return Eigen::Isometry3d::Identity();
            }
        }

        // Otherwise, just return the most recent pose
        return this->kf_poses_updated_.back();
    }
    
}

/**
 * @brief Gets the index of the current keyframe.
 */
int PoseGraphManager::currentKFIndex() const
{
    std::lock_guard<std::mutex> lock(this->kf_mtx_);
    return this->kf_poses_odom_.size() - 1;
}

/**
 * @brief Gets the size (number of keyframes) of the pose graph.
 */
int PoseGraphManager::graphSize() const
{
    std::lock_guard<std::mutex> lock(this->kf_mtx_);
    return this->kf_poses_odom_.size();
}

/**
 * @brief Returns the vector of updated poses
 */
std::vector<Eigen::Isometry3d> PoseGraphManager::getUpdatedKFPoses() const
{
    std::lock_guard<std::mutex> lock(this->kf_updated_mtx_);
    return this->kf_poses_updated_;
}

/**
 * @brief Returns a vector of GNSS points
 */
std::vector<std::optional<Eigen::Vector3d>> PoseGraphManager::getGNSSPoints() const
{
    std::lock_guard<std::mutex> lock(this->kf_mtx_);
    return this->kf_gnss_;
}

/**
 * @brief returns a vector of the keyframe timestamps
 */
std::vector<double> PoseGraphManager::getKFTimestamps() const
{
    std::lock_guard<std::mutex> lock(this->kf_mtx_);
    return this->kf_timestamps_;
}

/**
 * @brief Returns a keyframe's point cloud at the given index.
 * @param index The keyframe index.
 * @return The point cloud at the given index, or nullptr if out of bounds.
 */
pcl::PointCloud<PointType>::Ptr PoseGraphManager::getKeyframePointCloud(int index) const
{
    std::lock_guard<std::mutex> lock(this->kf_mtx_);
    if (index < 0 || index >= static_cast<int>(this->kf_pointclouds.size())) {
        return nullptr;
    }
    return this->kf_pointclouds.at(index);
}

/**
 * @brief Triggers extra optimization on the pose graph at next optimization
 */
void PoseGraphManager::triggerExtraOptimization()
{
    this->trigger_extra_optimization_flag_ = true;
}

/**
 * @brief Sets the current pose of the pose graph manager (with mutex)
 * 
 * @param pose The new current pose to be set.
 */
void PoseGraphManager::setCurrentPose(const Eigen::Isometry3d &pose)
{
    std::lock_guard<std::mutex> lock(this->current_pose_mtx_);
    this->T_curr_updated_ = pose;
}

/**
 * @brief Sets the loop closure manager.
 * @param lc_manager The loop closure manager to set.
 */
void PoseGraphManager::setLoopClosureManager(std::shared_ptr<LoopClosureManager> lc_manager)
{
    this->loop_closure_manager_ = lc_manager;
}

/**
 * @brief Adds a prior factor to the pose graph at a specified keyframe index.
 * @param T The transformation matrix representing the prior pose.
 * @param kf_id The keyframe index for the prior factor. If -1, uses current graph size.
 */
void PoseGraphManager::addPriorFactorandEstimate(const Eigen::Isometry3d &T, int kf_id)
{
    gtsam::Pose3 pose(T.matrix());

    auto rotation_noise = this->params->useExternalOrientation() ?
        this->params->graph.orientation_noise :
        this->params->graph.prior_noise_rot;

    gtsam::noiseModel::Diagonal::shared_ptr noise =
    gtsam::noiseModel::Diagonal::Sigmas((Eigen::VectorXd(6) <<
        rotation_noise,
        rotation_noise,
        rotation_noise,
        this->params->graph.prior_noise_lin,
        this->params->graph.prior_noise_lin,
        this->params->graph.prior_noise_lin).finished()); // rad, meter

    // If kf_id is -1, use index 0 for backward compatibility
    if (kf_id < 0) {
        kf_id = 0;
    }

    {
        std::lock_guard<std::mutex> lock(this->graph_mtx_);
        this->graph.add(gtsam::PriorFactor<gtsam::Pose3>(kf_id, pose, noise));

        // Add keyframe estimate
        this->kf_isam_estimates_.insert(kf_id, gtsam::Pose3(T.matrix()));
    }

    ROS_INFO("Added prior factor at keyframe %d", kf_id);
}


/**
 * @brief Adds a GNSS factor to the pose graph at a specified keyframe index, and adds the estimate.
 * @param kf_index The keyframe index for the GNSS factor.
 * @param gnss_odom The GNSS odometry message.
 * @param T The transformation matrix representing the prior pose.
 */
void PoseGraphManager::addGNSSFactorAndEstimate(int kf_index, nav_msgs::Odometry::ConstPtr gnss_odom, const Eigen::Isometry3d &T)
{
    // Add estimate
    {
        std::lock_guard<std::mutex> lock(this->graph_mtx_);
        this->kf_isam_estimates_.insert(kf_index, gtsam::Pose3(T.matrix()));
    }

    // Force add GNSS factor
    this->addGNSSFactor(kf_index, gnss_odom, true);  // force_add = true
}

/**
 * @brief Adds an odometry factor to the pose graph.
 * 
 * @param T_delta The transformation matrix representing the relative pose between two keyframes.
 */
void PoseGraphManager::addOdometryFactorAndEstimate(int kf_index, const Eigen::Isometry3d &T_delta, Eigen::Isometry3d &T_est)
{
    if(kf_index < 1){
        ROS_ERROR("Invalid KF index to add odometry factor: %d", kf_index);
        return;
    }

    gtsam::Pose3 pose_between(T_delta.matrix());

    auto odometry_noise = gtsam::noiseModel::Diagonal::Sigmas( (Eigen::VectorXd(6) <<
        this->params->graph.lio_noise_rot,
        this->params->graph.lio_noise_rot,
        this->params->graph.lio_noise_rot,
        this->params->graph.lio_noise_lin,
        this->params->graph.lio_noise_lin,
        this->params->graph.lio_noise_lin
    ).finished());

    {
        // Add factor
        std::lock_guard<std::mutex> lock(this->graph_mtx_);
        this->graph.add(gtsam::BetweenFactor<gtsam::Pose3>(kf_index-1, kf_index, pose_between, odometry_noise));

        // Add estimate for new keyframe
        this->kf_isam_estimates_.insert(kf_index, gtsam::Pose3(T_est.matrix()));
    }
}

/**
 * @brief Estimates the updated pose based on the odometry data.
 * 
 * @param T_odom The odometry transformation matrix.
 * @return The estimated updated pose.
 */
Eigen::Isometry3d PoseGraphManager::estimateUpdatedPose(Eigen::Isometry3d T_odom) const
{
    // Get most recent KF pose
    Eigen::Isometry3d T_updated = this->updatedPose(false);
    int updated_index = this->updatedKFIndex();

    // Get odometry info at the updated pose (need to handle the case where graph hasn't been updated)
    Eigen::Isometry3d T_odom_old;
    if(updated_index < 0 || updated_index >= static_cast<int>(this->kf_poses_odom_.size())){
        if(this->graphInitialized()) T_odom_old = this->T0_;
        else T_odom_old = Eigen::Isometry3d::Identity();
    }else{
        T_odom_old = this->kf_poses_odom_.at(updated_index);
    }

    // Compute delta pose between current and old odometry
    Eigen::Isometry3d T_delta = T_odom_old.inverse() * T_odom;

    // Apply this delta to the most recent updated pose to get the estimate pose
    Eigen::Isometry3d T_est = T_updated * T_delta;

    return T_est;
}

/**
 * @brief Calculates the total distance traveled based on GNSS data.
 * 
 * @return The total distance traveled in meters.
 */
double PoseGraphManager::distanceTravelledGNSS_() const
{
    // Get GNSS poses (thread safe)
    auto kf_gnss = this->getGNSSPoints();

    double distance = 0;
    std::optional<Eigen::Vector3d> p_prev;
    for (const auto& p : kf_gnss){
        if(p.has_value()){
            if(p_prev.has_value()){
                distance += (p.value() - p_prev.value()).norm();
            }
            p_prev = p;
        }
    }
    return distance;
}

/**
 * @brief Adds a GNSS factor to the pose graph.
 * 
 * @param gnss_odom Pointer to a GNSS odometry message.
 */
void PoseGraphManager::addGNSSFactor(int kf_index, nav_msgs::Odometry::ConstPtr gnss_odom, bool force_add)
{
    // Break out pose
    auto gnss_pose = gnss_odom->pose;

    // If covariance is too high, exit
    if(gnss_pose.covariance[0] > this->params->graph.gps_noise_threshold || gnss_pose.covariance[7] > this->params->graph.gps_noise_threshold) return;

    // Discard non-positive variances (sometimes used when no fix); sqrt would give a NaN sigma
    if(gnss_pose.covariance[0] <= 0.0 || gnss_pose.covariance[7] <= 0.0 || gnss_pose.covariance[14] <= 0.0){
        ROS_WARN_THROTTLE(5, "Discarding GNSS factor at kf %d: non-positive position variance (%.3f, %.3f, %.3f)",
                          kf_index, gnss_pose.covariance[0], gnss_pose.covariance[7], gnss_pose.covariance[14]);
        return;
    }

    // Check if orientation is provided (non-zero orientation covariance indicates orientation is set)
    bool has_orientation = (gnss_pose.covariance[35] > 0);  // Check yaw variance

    // Log the GPS measurement position
    gtsam::Point3 gnss_pnt(gnss_pose.pose.position.x, gnss_pose.pose.position.y, gnss_pose.pose.position.z);
    {
        std::lock_guard<std::mutex> lock(this->kf_mtx_);
        this->kf_gnss_.resize(kf_index+1, std::nullopt);
        this->kf_gnss_.at(kf_index) = gnss_pnt.vector();
    }

    // Build the factor based on whether we have orientation
    gtsam::NonlinearFactor::shared_ptr factor;
    
    if (has_orientation) {
        // Full 6DOF pose factor
        gtsam::Rot3 gnss_rot(gtsam::Quaternion(
            gnss_pose.pose.orientation.w,
            gnss_pose.pose.orientation.x,
            gnss_pose.pose.orientation.y,
            gnss_pose.pose.orientation.z
        ));
        gtsam::Pose3 gnss_pose3(gnss_rot, gnss_pnt);

        // Build 6DOF noise model (rot_x, rot_y, rot_z, x, y, z)
        gtsam::Vector6 noise_sigmas;
        
        // Rotation noise (from covariance diagonal, take sqrt for sigma)
        noise_sigmas(0) = std::sqrt(gnss_pose.covariance[21]);  // roll
        noise_sigmas(1) = std::sqrt(gnss_pose.covariance[28]);  // pitch
        noise_sigmas(2) = std::sqrt(gnss_pose.covariance[35]);  // yaw

        // Position noise
        noise_sigmas(3) = std::sqrt(gnss_pose.covariance[0]) * this->params->graph.gps_noise_scale;   // x
        noise_sigmas(4) = std::sqrt(gnss_pose.covariance[7]) * this->params->graph.gps_noise_scale;   // y
        noise_sigmas(5) = std::sqrt(gnss_pose.covariance[14]) * this->params->graph.gps_noise_scale * this->params->graph.gps_noise_z_scale;  // z

        // Disable altitude constraint if configured
        if (!this->params->graph.use_gnss_altitude) {
            noise_sigmas(5) = 1e5;  // Very large uncertainty = effectively disabled
        }

        auto noise_model = gtsam::noiseModel::Diagonal::Sigmas(noise_sigmas);
        factor = boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(kf_index, gnss_pose3, noise_model);

        ROS_DEBUG("Adding GNSS+Orientation factor at kf %d: pos=(%.2f, %.2f, %.2f), has orientation", 
                  kf_index, gnss_pnt.x(), gnss_pnt.y(), gnss_pnt.z());
    } else {
        // Position-only GPS factor (original behavior)
        gtsam::Vector3 gnss_noise;
        gnss_noise << gnss_pose.covariance[0], gnss_pose.covariance[7], gnss_pose.covariance[14];
        gnss_noise *= (this->params->graph.gps_noise_scale * this->params->graph.gps_noise_scale);
        gnss_noise(2) *= (this->params->graph.gps_noise_z_scale * this->params->graph.gps_noise_z_scale);

        if (!this->params->graph.use_gnss_altitude) {
            gnss_noise(2) = 1e5; // Very large uncertainty = effectively disabled
        }

        auto gnss_noise_model = gtsam::noiseModel::Diagonal::Variances(gnss_noise);
        factor = boost::make_shared<gtsam::GPSFactor>(kf_index, gnss_pnt, gnss_noise_model);

        ROS_DEBUG("Adding GPS factor (position only) at kf %d: pos=(%.2f, %.2f, %.2f)", 
                  kf_index, gnss_pnt.x(), gnss_pnt.y(), gnss_pnt.z());
    }

    // Add GNSS factor to graph
    {
        std::lock_guard<std::mutex> lock(this->graph_mtx_);

        // Determine whether to add immediately:
        // - Already initialized: always add immediately
        // - force_add: caller requested immediate addition (e.g., session initialization)
        // - External orientation enabled: no need for delayed initialization since we have heading
        bool add_immediately = this->gnss_initialized_ || force_add || this->params->useExternalOrientation();

        if (add_immediately) {
            // Flush any buffered factors first (if not yet initialized)
            if (!this->gnss_initialized_) {
                ROS_INFO("Initializing GNSS transform!");
                while (!this->gnss_factor_buffer_.empty()) {
                    this->graph.add(this->gnss_factor_buffer_.front());
                    this->gnss_factor_buffer_.pop();
                }
                this->gnss_initialized_ = true;
                this->triggerExtraOptimization();
            }

            this->graph.add(factor);
            ROS_DEBUG("Logging GPS measurement (to graph), kf %d: x = %.2f, y = %.2f, z = %.2f.", kf_index, gnss_pnt.x(), gnss_pnt.y(), gnss_pnt.z());
        } else {
            // Delayed adding: check if we've travelled far enough to initialize GNSS
            if (this->distanceTravelledGNSS_() > this->params->graph.gnss_min_initialization_distance) {
                // Distance travelled threshold is satisfied - initialize GNSS
                ROS_INFO("Initializing GNSS transform!");
                while (!this->gnss_factor_buffer_.empty()) {
                    this->graph.add(this->gnss_factor_buffer_.front());
                    this->gnss_factor_buffer_.pop();
                }

                this->graph.add(factor);
                this->gnss_initialized_ = true;
                this->triggerExtraOptimization();
            } else {
                // Just add the GNSS factor to the queue
                this->gnss_factor_buffer_.push(factor);
                ROS_DEBUG("Logging GPS measurement (to queue), kf %d: x = %.2f, y = %.2f, z = %.2f. Distance travelled: %.2f", kf_index, gnss_pnt.x(), gnss_pnt.y(), gnss_pnt.z(), this->distanceTravelledGNSS_());
            }
        }
    }
}

/**
 * * @brief Adds a loop closure factor to the pose graph.
 * @param kf_prev The index of the first keyframe.
 * @param kf_index_2 The index of the second keyframe.
 * @param Ticp The transformation matrix representing the loop closure.
 */
void PoseGraphManager::addLoopClosureFactor(int kf_prev, int kf_curr, Eigen::Isometry3d T_between, double fitnessScore)
{
    // Already the relative pose X_prev^-1 * X_curr_corrected, composed in icp() against the
    // pose snapshot the registration used.
    gtsam::Pose3 betweenPose = gtsam::Pose3(T_between.matrix());

    // Initialize noise vector
    gtsam::Vector robustNoiseVector6(6);
    robustNoiseVector6 << fitnessScore, fitnessScore, fitnessScore, fitnessScore, fitnessScore, fitnessScore;

    // Scale noise
    robustNoiseVector6 *= (this->params->graph.loop_closure_noise_scale * this->params->graph.loop_closure_noise_scale);

    // Build noise model
    auto loopNoise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Cauchy::Create(1),
        gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));

    // Add to graph
    {
        std::lock_guard<std::mutex> lock(this->graph_mtx_);
        this->graph.add(gtsam::BetweenFactor<gtsam::Pose3>(kf_prev, kf_curr, betweenPose, loopNoise));
    }
    this->triggerExtraOptimization();
}

/**
 * @brief Processes a data point by updating the pose graph with new odometry and point cloud data.
 *
 * This function handles the initialization and updating of the pose graph using odometry data,
 * point clouds, and optional GNSS data. It calculates the relative pose changes, manages keyframes,
 * and saves point cloud data. If the pose graph is not initialized, it sets up the initial pose and
 * adds a prior factor. For subsequent calls, it computes the pose change, checks if a new keyframe
 * should be added based on movement thresholds, accumulates point clouds, and updates the pose graph.
 *
 * @param data A DataPoint object containing odometry, point cloud, and optional GNSS data.
 */
void PoseGraphManager::processData(DataPoint data)
{

    // Note, need to change the frames to be in the same frame as the flight controller.
    // These transforms have been checked!!
    const auto& T_body_lidar = this->params->extrinsics.T_body_lidar;
    auto T_odom = data.T_odom * T_body_lidar.inverse();
    auto pointcloud = geometry::pclTransform(data.pointcloud, T_body_lidar.matrix());
    
    bool is_first_session = !this->graphInitialized();

    // Handle session initialization (either first session or after reinitializeSession())
    if (is_first_session || this->awaiting_session_init_) {

        // For multi-drone (non-first session), warn if GNSS/orientation disabled
        if (!is_first_session && (!this->params->useGNSS() || !this->params->useExternalOrientation())) {
            ROS_WARN_ONCE("Multi-drone mode without GNSS and external orientation enabled will likely produce poor results. "
                          "The new drone session will start at identity pose.");
        }

        // Wait for GPS data to initialize (if GNSS is enabled)
        if (this->params->useGNSS() && !data.gnss_msg.has_value()) {
            ROS_WARN_THROTTLE(5, "Waiting for a GNSS sample before initializing session.");
            return;
        }

        // Wait for orientation (if external orientation is enabled)
        if (this->params->useExternalOrientation() && !data.orientation_msg.has_value()) {
            ROS_WARN_THROTTLE(5, "Waiting for orientation message before initializing session.");
            return;
        }

        // Get first odometry pose (use this to convert to relative measurements later)
        this->T_odom_prev_ = T_odom;

        // Build the session start pose
        Eigen::Isometry3d T_session_start = Eigen::Isometry3d::Identity();

        // Set orientation from external source if available
        if (this->params->useExternalOrientation() && data.orientation_msg.has_value()) {
            T_session_start.linear() = data.orientation_msg.value().toRotationMatrix();
            ROS_INFO("Using external orientation for session start pose.");
        }

        // Set initial position from GNSS, if available (and set datum)
        if (this->params->useGNSS() && data.gnss_msg.has_value()) {
            // First session: start at origin, reset GNSS datum
            // Reset GNSS origin so first keyframe is at (0,0,0)
            if (is_first_session){
                this->navSatFixToOdometry(data.gnss_msg.value(), data.orientation_msg, true);
            } else {
                // Subsequent session: get position from GNSS in the existing coordinate frame
                // Convert GNSS to local coordinates (using existing origin, NOT resetting)
                auto gnss_odom = this->navSatFixToOdometry(data.gnss_msg.value(), data.orientation_msg, false);
                T_session_start.translation() = Eigen::Vector3d(
                    gnss_odom->pose.pose.position.x,
                    gnss_odom->pose.pose.position.y,
                    gnss_odom->pose.pose.position.z
                );
            }
        }

        // Add keyframe (index determined by current kf_poses_odom_ size)
        this->T0_ = T_session_start;
        int new_kf_index = this->addKeyframe(T_session_start, pointcloud, data.timestamp_lio);

        // Anchor the session with a fixed-noise prior, in every case.
        //
        // T_session_start already carries everything a GNSS-derived factor would have
        // supplied: the external orientation (set above, when available) and the GNSS
        // position -- exactly (0,0,0) for the first session because the datum was just
        // reset to this fix, and the converted position in the existing frame for every
        // session after it. So there is nothing left for a separate GNSS factor to add
        // here, and no reason to branch.
        //
        // Using the prior instead of addGNSSFactorAndEstimate also fixes the conditioning.
        // The GNSS factor derives its sigmas from the fix covariance scaled by
        // gps_noise_scale (and gps_noise_z_scale vertically), which on a below-canopy fix
        // is metres before scaling and tens of metres after. That is loose enough to leave
        // the session's first pose effectively unconstrained, and iSAM2 fails with
        // IndeterminantLinearSystemException on that variable. The prior uses
        // prior_noise_lin, and prior_noise_rot / orientation_noise for rotation, which are
        // fixed, moderate, and independent of whatever the receiver reports.
        this->addPriorFactorandEstimate(T_session_start, new_kf_index);

        // Clear awaiting flag
        this->awaiting_session_init_ = false;

        ROS_INFO("Initialized session %d at keyframe %d: %s",
                 this->current_session_id_, new_kf_index, isometryToStr(T_session_start).c_str());
        return;
    }

    // Compute current pose
    Eigen::Isometry3d T_delta = this->T_odom_prev_.inverse() * T_odom;

    // Compute the "estimated" pose from LIO
    Eigen::Isometry3d T_kf = this->kf_poses_odom_.back() * T_delta;
    Eigen::Isometry3d T_kf_updated = this->estimateUpdatedPose(T_kf);
    this->setCurrentPose(T_kf_updated);

    // Early reject by counting local delta movement (for equi-spreated kf drop)
    // Compute distance travelled since previous odometry keyframe
    auto [linear_distance, angular_distance] = geometry::poseDistance(this->T_odom_prev_, T_odom);

    // Break out of loop if distance travelled is insufficient. Otherwise, reset accumulated travel.
    if( linear_distance < this->params->graph.kf_gap_lin && angular_distance < this->params->graph.kf_gap_rot ){
        // if skipping, save the point cloud data and transform
        this->inter_kf_pointcloud_buffer_.push({T_odom, pointcloud});
        return;
    }
                
    // Accumulate previous pointclouds into the current frame while transforming
    while(!this->inter_kf_pointcloud_buffer_.empty()){
        const auto& T_i = this->inter_kf_pointcloud_buffer_.front().first;
        auto pointcloud_i = this->inter_kf_pointcloud_buffer_.front().second;
        *pointcloud += * geometry::pclTransform(pointcloud_i, (T_odom.inverse() * T_i ).matrix());
        this->inter_kf_pointcloud_buffer_.pop();
    }

    // Store downsampled data to vectors of laser clouds, poses, and data.timestamp_lio
    int current_kf_index = this->addKeyframe(T_kf, pointcloud, data.timestamp_lio);

    // Reset previous odometry variable
    this->T_odom_prev_ = T_odom;
    
    // Get odometry poses and noise
    this->addOdometryFactorAndEstimate(current_kf_index, T_delta, T_kf_updated);

    // Add GNSS, if one is available    
    if (this->params->useGNSS() && data.gnss_msg.has_value()) {
        // Pass orientation if available
        auto gnss_odom = this->navSatFixToOdometry(
            data.gnss_msg.value(), 
            data.orientation_msg,  // Pass current orientation (may be nullopt)
            false  // resetOrigin
        );
        this->addGNSSFactor(current_kf_index, gnss_odom);
    }

    // Save point cloud
    this->savePointCloud(current_kf_index, pointcloud, data.timestamp_laser);
}

void PoseGraphManager::optimizeGraph()
{
    // Exit if there is no graph
    if(!this->graphInitialized()) return;
    {
        std::lock_guard<std::mutex> lock(this->graph_mtx_);
        this->isam_.update(this->graph, this->kf_isam_estimates_);

        // Clear the graph since all the factors have already been added to ISAM optimizer
        this->graph.resize(0);
        this->kf_isam_estimates_.clear();
    }

    // Add all new graph factors to the ISAM optimizer
    this->isam_.update();

    if (this->trigger_extra_optimization_flag_) {
        for(int i=0; i<5; i++) this->isam_.update();
    }
    this->trigger_extra_optimization_flag_ = false;


    // Get updates
    auto isam_estimates_updated = this->isam_.calculateEstimate();
    
    // Update keyframes
    {
        std::lock_guard<std::mutex> lock(this->kf_updated_mtx_);

        size_t N = isam_estimates_updated.size();
        this->kf_poses_updated_.resize(N);

        for(size_t i=0; i < N; i++){
            this->kf_poses_updated_.at(i) = Eigen::Isometry3d(isam_estimates_updated.at<gtsam::Pose3>(i).matrix());
        }
    }

    // Save the frames positions
    this->saveGraphKeyframes();
}

/**
 * @brief Saves a point cloud to a file and logs its timestamp.
 *
 * @param pointcloud Pointer to the point cloud to be saved.
 * @param timestamp The timestamp associated with the point cloud, used for logging.
 */
void PoseGraphManager::savePointCloud(int kf_index, pcl::PointCloud<PointType>::Ptr pointcloud, double timestamp) const
{
    std::string filename = this->params->ros.save_directory + "scans/" + padZeros(kf_index) + ".pcd";
    
    ROS_DEBUG("Saving point cloud to file: %s", filename.c_str());

    if (pcl::io::savePCDFileBinary(filename, *pointcloud) == -1) {
        ROS_ERROR("Failed to save point cloud file: %s", filename.c_str());
    }

    // Write the timestamp to the associated time stream
    this->pg_laser_timestamp_save_ << timestamp << std::endl;
}

/**
 * @brief Saves the current, updated poses to a file
 */
void PoseGraphManager::saveGraphKeyframes() const
{
    if(!this->graphInitialized()) return;

    std::string filename_kf = this->params->ros.save_directory + "optimized_poses.txt";
    saveTransformsToFile(this->getUpdatedKFPoses(), this->kf_session_ids_, filename_kf);

    // The pre-optimisation poses, in the same keyframe index space. These are the FAST-LIO
    // odometry poses as they were when each keyframe was created, before any loop closure
    // or iSAM2 update touched them -- within a session, that is pure dead reckoning; across
    // sessions each start pose is still anchored by its first GNSS fix. Differencing the two
    // files gives the drift the pose graph removed, which is otherwise unrecoverable once
    // the run is over.
    std::string filename_odom = this->params->ros.save_directory + "odometry_poses.txt";
    saveTransformsToFile(this->kf_poses_odom_, this->kf_session_ids_, filename_odom);

    if(!this->gnss_origin_) return;

    std::string filename_gnss_origin = this->params->ros.save_directory + "gnss_origin.txt";
    std::fstream stream(filename_gnss_origin.c_str(), std::fstream::out);
    // Full precision: the default ostream precision (6 significant figures) truncates
    // latitude/longitude to ~5.5 m N-S and ~2.8 m E-W, which is coarser than the
    // stem-level accuracy the maps are used for.
    // std::fixed keeps trailing zeros, so the decimal count always reflects true precision.
    stream << std::fixed << std::setprecision(9)
           << "latitude: " << this->gnss_origin_.value()(0)
           << " longitude: " << this->gnss_origin_.value()(1)
           << " altitude: " << this->gnss_origin_.value()(2) << std::endl;
}

/**
 * @brief Assemble a global pointcloud using the updated pose estimates
 *
 * @param frame_skip will skip keyframes to reduce computational intensity
 */
pcl::PointCloud<PointType>::Ptr PoseGraphManager::assembleGlobalPointCloud(int frame_skip) const {

    pcl::PointCloud<PointType>::Ptr globalCloud(new pcl::PointCloud<PointType>());
    auto updatedPoses = this->getUpdatedKFPoses();

    for (size_t i = 0; i < updatedPoses.size(); i += frame_skip) {
        auto pointcloud = this->getKeyframePointCloud(i);
        if (pointcloud) {
            *globalCloud += *geometry::pclTransform(pointcloud, updatedPoses.at(i).matrix());
        }
    }

    return globalCloud;
}

/**
 * @brief Converts a NavSatFix message to an Odometry message.
 *
 * This function takes a GNSS (NavSatFix) message and converts it into an Odometry message.
 * If the GNSS origin is not already set (i.e. gnss_origin_ is empty), the function resets the internal geoConverter
 * using the latitude, longitude, and altitude from the NavSatFix message and stores this as the GNSS origin.
 *
 * @param nav_sat_fix A shared pointer to the constant NavSatFix message containing GNSS data.
 * @param orientation An optional quaternion representing the orientation to be used in the Odometry message.
 * @param resetOrigin A boolean flag indicating whether to reset the GNSS origin.
 * @return nav_msgs::Odometry::ConstPtr A constant shared pointer to the resulting Odometry message.
 */
nav_msgs::Odometry::ConstPtr PoseGraphManager::navSatFixToOdometry(
    const sensor_msgs::NavSatFix::ConstPtr& nav_sat_fix,
    std::optional<Eigen::Quaterniond> orientation, 
    bool resetOrigin)
{
    if (!this->gnss_origin_.has_value() || resetOrigin)
    {
        this->geo_converter_.Reset(nav_sat_fix->latitude, nav_sat_fix->longitude, nav_sat_fix->altitude);
        this->gnss_origin_ = Eigen::Vector3d(nav_sat_fix->latitude, nav_sat_fix->longitude, nav_sat_fix->altitude);
        ROS_INFO("Reset local coordinate frame origin to: lat: %.2f, long: %.2f, altitude: %.2f m.", 
                 nav_sat_fix->latitude, nav_sat_fix->longitude, nav_sat_fix->altitude);
    }

    nav_msgs::Odometry odom;   
    odom.header.frame_id = "map";
    odom.header.stamp = nav_sat_fix->header.stamp;

    // Set position
    this->geo_converter_.Forward(nav_sat_fix->latitude, nav_sat_fix->longitude, nav_sat_fix->altitude, 
                                 odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z);

    // Set orientation - use provided quaternion or identity
    if (orientation.has_value()) {
        const auto& q = orientation.value();
        odom.pose.pose.orientation.w = q.w();
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
    } else {
        odom.pose.pose.orientation.w = 1.0;
        odom.pose.pose.orientation.x = 0.0;
        odom.pose.pose.orientation.y = 0.0;
        odom.pose.pose.orientation.z = 0.0;
    }

    // Position covariances from NavSatFix
    odom.pose.covariance[0] = nav_sat_fix->position_covariance[0];   // X variance
    odom.pose.covariance[7] = nav_sat_fix->position_covariance[4];   // Y variance
    odom.pose.covariance[14] = nav_sat_fix->position_covariance[8];  // Z variance

    // Orientation covariances - use fixed values when orientation is provided
    // These go in indices 21 (roll), 28 (pitch), 35 (yaw) for a 6x6 covariance matrix
    if (orientation.has_value()) {
        // Set orientation uncertainty - tune these based on your orientation source
        odom.pose.covariance[21] = this->params->graph.orientation_noise*this->params->graph.orientation_noise;  // roll
        odom.pose.covariance[28] = this->params->graph.orientation_noise*this->params->graph.orientation_noise;  // pitch
        odom.pose.covariance[35] = this->params->graph.orientation_noise*this->params->graph.orientation_noise;  // yaw
    }

    // Zero out cross-correlations
    for (size_t i = 0; i < odom.pose.covariance.size(); ++i) {
        if (i != 0 && i != 7 && i != 14 && i != 21 && i != 28 && i != 35) {
            odom.pose.covariance[i] = 0.0;
        }
    }

    nav_msgs::Odometry::Ptr odom_ptr = boost::make_shared<nav_msgs::Odometry>(odom);
    return nav_msgs::Odometry::ConstPtr(odom_ptr);
}

/**
 * @brief Re-initializes the PGO session for a new session in a multi-session run.
 *
 * Advances the session ID, clears the data buffer, and sets the flag to await
 * session initialization on the next processData call.
 * The ScanContext database is preserved to enable inter-session loop closures.
 *
 * @return The next keyframe index that will be used for the new session.
 */
int PoseGraphManager::reinitializeSession()
{
    // Only advance if the current session actually produced something, so the orchestrator can
    // call reinit before every bag (including the first) without leaving a gap in the ids
    if (!this->kf_poses_odom_.empty()) this->current_session_id_++;

    int next_kf_index = this->kf_poses_odom_.size();

    this->data_buffer.clearAll();

    // Clear the inter-keyframe buffer
    while (!this->inter_kf_pointcloud_buffer_.empty()) {
        this->inter_kf_pointcloud_buffer_.pop();
    }

    // Reset odometry tracking for new session
    this->T_odom_prev_ = Eigen::Isometry3d::Identity();

    this->awaiting_session_init_ = true;

    ROS_INFO("Re-initialized for session %d. Next keyframe index: %d",
             this->current_session_id_, next_kf_index);

    return next_kf_index;
}

/**
 * @brief Returns the current session ID.
 */
uint8_t PoseGraphManager::getCurrentSessionId() const
{
    return this->current_session_id_;
}

/**
 * @brief Returns the session ID for a specific keyframe.
 * @param kf_index The keyframe index.
 * @return The session ID, or 0 if index is out of bounds.
 */
uint8_t PoseGraphManager::getKeyframeSessionId(int kf_index) const
{
    std::lock_guard<std::mutex> lock(this->kf_mtx_);
    if (kf_index < 0 || kf_index >= static_cast<int>(this->kf_session_ids_.size())) {
        return 0;
    }
    return this->kf_session_ids_.at(kf_index);
}

/**
 * @brief Returns the session ID associated with each keyframe.
 */
std::vector<uint8_t> PoseGraphManager::getKeyframeSessionIds() const
{
    std::lock_guard<std::mutex> lock(this->kf_mtx_);
    return this->kf_session_ids_;
}

/**
 * @brief Returns session boundary information as pairs of (start_index, session_id).
 */
std::vector<std::pair<int, uint8_t>> PoseGraphManager::getSessionBoundaries() const
{
    std::lock_guard<std::mutex> lock(this->kf_mtx_);

    std::vector<std::pair<int, uint8_t>> boundaries;

    if (this->kf_session_ids_.empty()) {
        return boundaries;
    }

    // First session always starts at index 0
    boundaries.push_back({0, this->kf_session_ids_[0]});

    // Find where session ID changes
    for (size_t i = 1; i < this->kf_session_ids_.size(); ++i) {
        if (this->kf_session_ids_[i] != this->kf_session_ids_[i - 1]) {
            boundaries.push_back({static_cast<int>(i), this->kf_session_ids_[i]});
        }
    }

    return boundaries;
}

/**
 * @brief Returns whether GNSS has been initialized.
 */
bool PoseGraphManager::isGNSSInitialized() const
{
    return this->gnss_initialized_;
}