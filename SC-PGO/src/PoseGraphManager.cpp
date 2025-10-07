#include "spade_pgo/PoseGraphManager.hpp"

#include <mutex>
#include <queue>
#include <vector>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

// SPADE PGO includes
#include "spade_pgo/common.hpp"
#include "spade_pgo/geometry.hpp"
#include "spade_pgo/PGOParams.hpp"
#include "spade_pgo/OrientationInitializer.hpp"
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
// #include <GeographicLib/UTMUPS.hpp>
// #include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/LocalCartesian.hpp>
// #include <GeographicLib/Geoid.hpp>

#include "scancontext/Scancontext.h"

using namespace spade_pgo;

/**
 * @brief Constructs a PoseGraphManager object with the given parameters.
 * 
 * @param params Shared pointer to the parameters object containing configuration settings.
 */
PoseGraphManager::PoseGraphManager(
    std::shared_ptr<PGOParams> params
) : params(params),
    orienter(std::make_unique<OrientationInitializer>(params->graph.orientation_calibration_size))
{
    // Build isam optimizer
    gtsam::ISAM2Params isam_params;
    isam_params.relinearizeThreshold = 0.01;
    isam_params.relinearizeSkip = 1;
    this->isam_ = gtsam::ISAM2(isam_params);

    // Setup voxel filters
    this->voxelizer_sc_.setLeafSize(this->params->sc.voxel_size, this->params->sc.voxel_size, this->params->sc.voxel_size);
    double voxel_size_save = std::min(this->params->icp.voxel_size, this->params->sc.voxel_size);
    this->voxelizer_save_.setLeafSize(voxel_size_save, voxel_size_save, voxel_size_save);

    // Setup outputs
    this->pg_laser_timestamp_save_ = std::fstream(this->params->ros.save_directory + "laser_timestamps.txt", std::fstream::out); 
    this->pg_laser_timestamp_save_.precision(std::numeric_limits<double>::max_digits10);
}

/**
 * @brief Adds a keyframe to the pose graph (consisting of, at a minimum, an odometry transform and pointcloud)
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
    }
    kf_id = this->kf_poses_odom_.size() - 1;

    ROS_DEBUG("Added keyframe %d, estimated position: %s", kf_id, isometryToStr(T).c_str());

    // Add to scancontext
    // TO DO: Fix this scancontrol stuff
    if(this->params->loop_closure.use_scancontrol){
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
    return this->kf_poses_odom_.size() - 1;
}

/**
 * @brief Gets the size (number of keyframes) of the pose graph.
 */
int PoseGraphManager::graphSize() const
{
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
 * @brief Adds a prior factor to the pose graph.
 * 
 * @param T The transformation matrix representing the prior pose.
 */
void PoseGraphManager::addPriorFactorandEstimate(const Eigen::Isometry3d &T)
{
    gtsam::Pose3 pose(T.matrix());

    gtsam::noiseModel::Diagonal::shared_ptr noise =
    gtsam::noiseModel::Diagonal::Sigmas((Eigen::VectorXd(6) <<
        this->params->graph.prior_noise_rot,
        this->params->graph.prior_noise_rot,
        this->params->graph.prior_noise_rot,
        this->params->graph.prior_noise_lin,
        this->params->graph.prior_noise_lin,
        this->params->graph.prior_noise_lin).finished()); // rad, meter
    
    const int kf_id = 0; // Initial pose
    {
        std::lock_guard<std::mutex> lock(this->graph_mtx_);
        this->graph.add(gtsam::PriorFactor<gtsam::Pose3>(kf_id, pose, noise));

        // Add keyframe estimate
        this->kf_isam_estimates_.insert(kf_id, gtsam::Pose3(T.matrix()));
    }
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
void PoseGraphManager::addGNSSFactor(int kf_index, nav_msgs::Odometry::ConstPtr gnss_odom)
{
    // Declare static variable
    static nav_msgs::Odometry::ConstPtr last_gps = nullptr;

    // Break out pose
    auto gnss_pose = gnss_odom->pose;

    // If covariance is too high, exit
    if(gnss_pose.covariance[0] > this->params->graph.gps_noise_threshold || gnss_pose.covariance[7] > this->params->graph.gps_noise_threshold) return;

    last_gps = gnss_odom;

    // Build gtsam pose and noise
    gtsam::Point3 gnss_pnt(gnss_pose.pose.position.x, gnss_pose.pose.position.y, gnss_pose.pose.position.z);
    gtsam::Vector gnss_noise(3);
    gnss_noise << gnss_pose.covariance[0], gnss_pose.covariance[7], gnss_pose.covariance[14];
    gnss_noise *= (this->params->graph.gps_noise_scale*this->params->graph.gps_noise_scale); // Scale by square of scaling factor
    gnss_noise(2) *= (this->params->graph.gps_noise_z_scale*this->params->graph.gps_noise_z_scale); // Further scale the z-factor by the scaling factor squared.

    // Adjust noise on z-vector if not using GPS altitude
    if( !this->params->graph.use_gnss_altitude ) gnss_noise(2) = 1e10; // OR 1e10

    // At first iteration, artificially reduce the gps noise to lock the map in place.
    // if(!this->gnss_initialized_ && this->gnss_factor_buffer_.size() == 0){
    //     gnss_noise *= 1e-6;
    // }

    // Build gps factor
    gtsam::noiseModel::Diagonal::shared_ptr gnss_noise_model = gtsam::noiseModel::Diagonal::Variances(gnss_noise);

    // Log the GPS measurement
    {
        std::lock_guard<std::mutex> lock(this->kf_mtx_);
        this->kf_gnss_.resize(kf_index+1, std::nullopt);
        this->kf_gnss_.at(kf_index) = gnss_pnt.vector();
    }

    // Make factor
    auto gnss_factor = gtsam::GPSFactor(kf_index, gnss_pnt, gnss_noise_model);

    // Add GNSS factor to graph
    {
        std::lock_guard<std::mutex> lock(this->graph_mtx_);

        if(this->gnss_initialized_){
            // GNSS is already initalized, just add it normally
            this->graph.add(gnss_factor);
            ROS_DEBUG("Logging GPS measurement (to graph), kf %d: x = %.2f, y = %.2f, z = %.2f.", kf_index, gnss_pnt.x(), gnss_pnt.y(), gnss_pnt.z());

        }else{
            // Check if we've travelled far enough to initialize GNSS
            if(this->distanceTravelledGNSS_() > this->params->graph.gnss_min_initialization_distance){
                // Distance travelled threshold is satisfied
                // Initialize GNSS
                ROS_INFO("Initializing GNSS transform!");
                while(!this->gnss_factor_buffer_.empty()){
                    this->graph.add(this->gnss_factor_buffer_.front());
                    this->gnss_factor_buffer_.pop();
                }
                
                this->gnss_initialized_ = true;
                this->triggerExtraOptimization();
            }else{
                // Just add the GNS Factor to the queue
                this->gnss_factor_buffer_.push(gnss_factor);
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
void PoseGraphManager::addLoopClosureFactor(int kf_prev, int kf_curr, Eigen::Isometry3d T_icp, double fitnessScore)
{
    gtsam::Pose3 betweenPose = gtsam::Pose3(T_icp.inverse().matrix());

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
    static Eigen::Isometry3d T_odom_prev;
    Eigen::Isometry3d& T_odom = data.T_odom;
    auto pointcloud = data.pointcloud;

    // If it's the first time this is run, set the "first" pose.
    if(! this->graphInitialized()){
        // We need to try initialize the graph.
        // We can only do this if: the calibration buffer is full, and
        // the current pose has a GPS pose.
        if(this->params->graph.use_gnss && !data.gnss_msg.has_value() ){
            ROS_WARN_THROTTLE(5, "Waiting for a GNSS sample before initializing graph.");
            return;
        }

        // Get first odometry pose (use this to convert to relative measurements later)
        T_odom_prev = T_odom;

        // Make the "first pose" from the orientation and position
        this->T0_ = Eigen::Isometry3d::Identity();
        if(this->params->graph.use_orientation_calibration){
            auto orientationOptional = this->orienter->getOrientation();
            if(! orientationOptional.has_value()){
                ROS_WARN_THROTTLE(5, "Waiting for OrientationInitilizer buffer to fill before initializing graph.");
                return;
            }
            this->T0_.linear() = orientationOptional.value();
        } // Otherwise, just use the identy orientation
        

        // Add keyframe to queue
        this->addKeyframe(this->T0_, pointcloud, data.timestamp_lio);

        // Add prior factor
        this->addPriorFactorandEstimate(this->T0_);

        // Add GNSS and reset origin
        if(this->params->graph.use_gnss) this->navSatFixToOdometry(data.gnss_msg.value(), true);
        // this->addGNSSFactor(0, gnss_odom); // no need to add to graph, it will just be 0,0,0 (same as prior factor).

        ROS_INFO("Initialized graph with prior factor: %s", isometryToStr(this->T0_).c_str());
        return;
    }

    // Compute current pose
    Eigen::Isometry3d T_delta = T_odom_prev.inverse() * T_odom;

    // Compute the "estimated" pose from LIO
    // Warning: May be a problem setting current pose so far from the addKeyframe part. Fix?
    Eigen::Isometry3d T_kf = this->kf_poses_odom_.back() * T_delta;
    Eigen::Isometry3d T_kf_updated = this->estimateUpdatedPose(T_kf);
    this->setCurrentPose(T_kf_updated);

    // Early reject by counting local delta movement (for equi-spreated kf drop)
    // Compute distance travelled since previous odometry keyframe
    auto [linear_distance, angular_distance] = geometry::poseDistance(T_odom_prev, T_odom);

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
    T_odom_prev = T_odom;
    
    // Get odometry poses and noise
    this->addOdometryFactorAndEstimate(current_kf_index, T_delta, T_kf_updated);

    // Add GNSS, if one is available    
    if (data.gnss_msg){
        auto gnss_odom = this->navSatFixToOdometry(data.gnss_msg.value());
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
    saveTransformsKITTI(this->getUpdatedKFPoses(), filename_kf);

    if(!this->gnss_origin_) return;

    std::string filename_gnss_origin = this->params->ros.save_directory + "gnss_origin.txt";
    std::fstream stream(filename_gnss_origin.c_str(), std::fstream::out);
    stream << "latitude: " << this->gnss_origin_.value()(0) << " longitude: " << this->gnss_origin_.value()(1) << " altitude: " << this->gnss_origin_.value()(2) << std::endl;
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
        *globalCloud += *geometry::pclTransform(this->kf_pointclouds.at(i), updatedPoses.at(i).matrix());
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
 * @return nav_msgs::Odometry::ConstPtr A constant shared pointer to the resulting Odometry message.
 */
nav_msgs::Odometry::ConstPtr PoseGraphManager::navSatFixToOdometry(const sensor_msgs::NavSatFix::ConstPtr& nav_sat_fix, bool resetOrigin)
{
    if (!this->gnss_origin_.has_value() || resetOrigin)
    {
        this->geo_converter_.Reset(nav_sat_fix->latitude, nav_sat_fix->longitude, nav_sat_fix->altitude);
        this->gnss_origin_ = Eigen::Vector3d(nav_sat_fix->latitude, nav_sat_fix->longitude, nav_sat_fix->altitude);
        ROS_INFO("Reset local coordinate frame origin to: lat: %.2f, long: %.2f, altitude: %.2f m.", nav_sat_fix->latitude, nav_sat_fix->longitude, nav_sat_fix->altitude);
    }

    // Create an Odometry message
    nav_msgs::Odometry odom;   
    odom.header.frame_id = "map";   // Set the frame ID
    odom.header.stamp = nav_sat_fix->header.stamp; // Use the timestamp from NavSatFix

    // Set the position in the Odometry message
    this->geo_converter_.Forward(nav_sat_fix->latitude, nav_sat_fix->longitude, nav_sat_fix->altitude, odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z);

    // Set an identity orientation if orientation information is not available
    odom.pose.pose.orientation.w = 1.0;

    // Copy covariances from NavSatFix
    // Assuming the covariances in NavSatFix are already in the correct units
    odom.pose.covariance[0] = nav_sat_fix->position_covariance[0];  // Variance in X (easting)
    odom.pose.covariance[7] = nav_sat_fix->position_covariance[4];  // Variance in Y (northing)
    odom.pose.covariance[14] = nav_sat_fix->position_covariance[8]; // Variance in Z (altitude)

    // Set all other covariances to zero (you can modify this if you have specific correlations or uncertainties)
    for (size_t i = 0; i < odom.pose.covariance.size(); ++i) {
        if (i != 0 && i != 7 && i != 14) {
            odom.pose.covariance[i] = 0.0;
        }
    }

    // Create a shared pointer to the Odometry object
    nav_msgs::Odometry::Ptr odom_ptr = boost::make_shared<nav_msgs::Odometry>(odom);

    // Return a ConstPtr from that
    return nav_msgs::Odometry::ConstPtr(odom_ptr);
}