#include <queue>
#include <vector>
#include <mutex>
#include <optional>
#include <utility>

#include "spade_pgo/LoopClosureManager.hpp"
#include "spade_pgo/geometry.hpp"
#include "spade_pgo/ros_helpers.hpp"

#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <small_gicp/registration/registration_helper.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/incremental_voxelmap.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/factors/icp_factor.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/ann/sequential_voxelmap_accessor.hpp>

#include <ros/ros.h>

namespace spade_pgo {

LoopClosureManager::LoopClosureManager(
    std::shared_ptr<PoseGraphManager> graph_manager,
    std::shared_ptr<Visualizer> visualizer,
    std::shared_ptr<PGOParams> params)
    :   near_kf_detector(params),
        graph_manager_(graph_manager),
        visualizer_(visualizer),
        params_(params)
{
    // Set up scancontext
    this->sc_detector.setSCdistThres(this->params_->sc.distance_threshold);
    this->sc_detector.setMaximumRadius(this->params_->sc.max_radius);
    this->voxelizer_sc_.setLeafSize(this->params_->sc.voxel_size, this->params_->sc.voxel_size, this->params_->sc.voxel_size);
}
LoopClosureManager::~LoopClosureManager() {}

/**
 * @brief Submits a loop closure candidate if it hasn't been tested yet.
 */
void LoopClosureManager::submitCandidate(int graph_id1, int graph_id2)
{
    // Round indicies to nearest entry in kf_indicies_
    int lc_id1 = this->findNearestLCKeyframe(graph_id1);
    int lc_id2 = this->findNearestLCKeyframe(graph_id2);
    int graph_id1_adj = this->kf_indicies_.at(lc_id1);
    int graph_id2_adj = this->kf_indicies_.at(lc_id2);

    // Check if pair already tested.
    for (const auto& p : this->tested_candidates_) {
        if ((p.first == lc_id1 && p.second == lc_id2) ||
            (p.first == lc_id2 && p.second == lc_id1)) {
            ROS_DEBUG("Already detected loop between %d and %d (graph ids: %d and %d)", lc_id1, lc_id2, graph_id1_adj, graph_id2_adj);
            return;
        }
    }

    // Add candidate in a thread-safe manner.
    {
        std::lock_guard<std::mutex> lock(this->candidate_mutex_);
        this->candidate_queue_.emplace(lc_id1, lc_id2);
    }

    // Add to list of tested candidates
    this->tested_candidates_.emplace_back(lc_id1, lc_id2);
    ROS_INFO("Added loop closure candidate between %d and %d (lc id: %d and %d). Initial detection: %d and %d", graph_id1_adj, graph_id2_adj, lc_id1, lc_id2, graph_id1, graph_id2);
}

void LoopClosureManager::updateCandidateQueue()
{
    // TO DO: Remove this when not debugging
    // Code to use when debugging to insert a loop closure between the first two frames.
    // This **should** converge, but normally you wouldn't want it there.
    static bool flag = false;

    if(this->kf_indicies_.size() > 2 && !flag){
        ROS_WARN("SUBMITTING TEST CANDIDATE BETWEEN 0 and 1: remove this code for disable!");
        flag = true;
        this->submitCandidate(this->kf_indicies_.at(0), this->kf_indicies_.at(1));
    }

    if(this->params_->loop_closure.use_near_kf){
        // Get and add near KF candidates
        auto candidates_near_kf = this->near_kf_detector.getNearKFCandidates(this->kf_indicies_, this->graph_manager_->getUpdatedKFPoses());

        for (const auto& candidate : candidates_near_kf){
            this->submitCandidate(candidate.first, candidate.second);
        }
    }

    // Only do SC if we have enough keyframes (and enabled)
    if( this->params_->loop_closure.use_scancontrol &&
        this->graph_manager_->graphSize() > this->sc_detector.NUM_EXCLUDE_RECENT)
    {
        auto sc_candidate = this->sc_detector.detectLoopClosureID();
        if (sc_candidate.first != -1)
            this->submitCandidate(sc_candidate.first, this->graph_manager_->currentKFIndex());
    }
}

void LoopClosureManager::processCandidateQueue()
{
    while ( !this->candidate_queue_.empty() )
    {
        if( this->candidate_queue_.size() > 200 ) {
            ROS_WARN_THROTTLE(2, "%ld loop closure candidates waiting... Adjust settings to produce fewer loop closure candidates", this->candidate_queue_.size());
        }

        // Pop the queue with a mutex for safety
        std::pair<int, int> candidate_pair;
        {
            std::lock_guard<std::mutex> lock(this->candidate_mutex_);
            candidate_pair = this->candidate_queue_.front();
            
            // We can't process loop closures until the pose has been updated by the graph.
            // Skip any that are not updated.
            if(this->kf_indicies_.at(candidate_pair.first) >= this->graph_manager_->updatedKFIndex() ||
               this->kf_indicies_.at(candidate_pair.second) >= this->graph_manager_->updatedKFIndex())
            {
                break;
            }
            
            this->candidate_queue_.pop();
        }

        // get BetweenFactor between poses (or std::nullopt if doesn't pass test)
        const int lc_id_prev = candidate_pair.first;
        const int lc_id_curr = candidate_pair.second;
        
        // Do ICP, add graph factor if valid
        if(auto relative_pose = this->icp(lc_id_prev, lc_id_curr)) {
            auto [Ticp, fitness_score] = relative_pose.value();

            // Add to list of loop closures to add to graph
            this->graph_manager_->addLoopClosureFactor(lc_id_prev, lc_id_curr, Ticp, fitness_score);

            // Add to list of added loop closures
            this->added_loop_closures_.emplace_back(this->kf_indicies_.at(lc_id_prev), this->kf_indicies_.at(lc_id_curr));
        } 
    }
}

void LoopClosureManager::submitKeyframeCloud(int kf_index, const Eigen::Isometry3d &T_i, pcl::PointCloud<PointType>::Ptr &pointcloud_i)
{
    // Initialize a prev transform to track distance
    static auto T_prev = Eigen::Isometry3d::Identity();

    // Figure out if we want to create a new keyframe
    if( geometry::euclideanDistance2(T_prev, T_i) >
    std::pow(this->params_->loop_closure.kf_distance, 2) ||
        this->kf_pointclouds_.empty() )
    {
        // Add voxel map to kf_pointclouds and id to kf_indices
        std::lock_guard<std::mutex> lock(this->kf_mutex_);
        auto new_voxelmap = std::make_shared<small_gicp::IncrementalVoxelMap<small_gicp::FlatContainerCov>>(this->params_->icp.voxel_size);
        new_voxelmap->lru_horizon = 1e9; // Disable LRU removal
        new_voxelmap->lru_clear_cycle = 1e9; // Disable LRU removal

        this->kf_indicies_.push_back(kf_index);
        this->kf_pointclouds_.push_back(new_voxelmap);

        ROS_DEBUG("Creating new LC keyframe at index %d.", kf_index);

        // Reset T_prev
        T_prev = T_i;

        // Publish point cloud to test visualization
        // size_t offset = 3;
        // if(this->kf_pointclouds_.size() >= offset){
        //     int lc_idx = this->kf_indicies_.size() - offset;
        //     int graph_idx = this->kf_indicies_.at(lc_idx);
        //     auto T_idx = this->graph_manager_->getKFCorrection(graph_idx);
        //     auto pointcloud_pcl = geometry::incrementalVoxelMapToPcl(*this->kf_pointclouds_.at(lc_idx), T_idx);
        //     this->visualizer_->publishTestCloud(pointcloud_pcl);
        // }
    }

    // Tranform the point clouds to small_gicp format
    auto eigen_cloud = geometry::pclToEigen(pointcloud_i);

    // Insert the new point cloud into the active voxel map (last entry) and, if available, into the previous one.
    {
        std::lock_guard<std::mutex> lock(this->kf_mutex_);
        // Active keyframe (last element)
        this->kf_pointclouds_.back()->insert(*eigen_cloud, T_i);

        // Optionally update the previous keyframe if needed.
        if(this->kf_pointclouds_.size() >= 2){
            this->kf_pointclouds_.at(this->kf_pointclouds_.size() - 2)->insert(*eigen_cloud, T_i);
        }
    }

    // Add to scancontext
    if(this->params_->loop_closure.use_scancontrol){
        // Downsample input point cloud from pointcloud according to voxel for SC
        pcl::PointCloud<PointType>::Ptr pointcloud_downsampled_sc(new pcl::PointCloud<PointType>());
        this->voxelizer_sc_.setInputCloud(pointcloud_i);
        this->voxelizer_sc_.filter(*pointcloud_downsampled_sc);
        this->sc_detector.makeAndSaveScancontextAndKeys(*pointcloud_downsampled_sc);
    }

}


/**
 * @brief Performs ICP to find the transformation between two keyframes.
 * @param lc_id_prev The index of the previous keyframe.
 * @param lc_id_curr The index of the current keyframe.
 * @return A pair containing the transformation and fitness score, or nullopt if ICP fails.
 */
std::optional<std::pair<Eigen::Isometry3d, double>> LoopClosureManager::icp( int lc_id_prev, int lc_id_curr )
{    
    // TO DO: Clean up the commented out code. 
    // Get pose corrections from graph
    Eigen::Isometry3d T_corr_prev = this->graph_manager_->getKFCorrection(this->kf_indicies_.at(lc_id_prev));
    Eigen::Isometry3d T_corr_curr = this->graph_manager_->getKFCorrection(this->kf_indicies_.at(lc_id_curr));

    // Get point clouds and voxelgrids
    // ros::Time pc_assy_time = ros::Time::now();
    auto voxelgrid_prev = this->kf_pointclouds_.at(lc_id_prev);
    auto voxelgrid_curr = this->kf_pointclouds_.at(lc_id_curr);
    const auto voxelgrid_accessor_curr = small_gicp::create_sequential_accessor(*voxelgrid_curr);

    // Can remove the pointcloud generation if can just use voxelgrids directly
    // small_gicp::PointCloud::Ptr pointcloud_prev = geometry::getPoints( voxelgrid_prev );
    small_gicp::PointCloud::Ptr pointcloud_curr = geometry::getPoints( voxelgrid_curr );  
    // Only for visualization
    // small_gicp::PointCloud::Ptr pointcloud_curr_viz = geometry::getPoints( voxelgrid_curr, (T_corr_curr.inverse() * T_corr_prev).inverse() );  
    // ROS_INFO("Point cloud assembly took %.3g ms.", (ros::Time::now() - pc_assy_time).toSec()*1e3);
    
    if(voxelgrid_curr->size() < 100 || voxelgrid_prev->size() < 100) {
        ROS_WARN("[Loop Closure] Point cloud size too small (%zu and %zu). Not adding loop closure between %d and %d.", voxelgrid_curr->size(), voxelgrid_prev->size(), lc_id_prev, lc_id_curr);
        return std::nullopt;
    }

    // Estimate covariances
    // ros::Time estimate_cov_time = ros::Time::now();
    // TO DO: Remove/clean up this
    small_gicp::estimate_covariances_omp(*voxelgrid_curr, 10, 6);
    // small_gicp::estimate_covariances_omp(*pointcloud_curr, 10, 6);
    // small_gicp::estimate_covariances_omp(*pointcloud_prev, 10, 6);
    // ROS_INFO("Covariance estimation took %.3g ms.", (ros::Time::now() - estimate_cov_time).toSec()*1e3);

    // Publish clouds, if in settings
    // TO DO: Re-enable this, but only when the parameters ask for it
    // ROS_INFO("Publishing loop closure clouds.");
    // this->visualizer_->publishLCClouds(pointcloud_curr_viz, pointcloud_prev);
    
    // Set up ICP settings
    ROS_INFO("Setting up registration.");
    small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP> registration;
    registration.reduction.num_threads = 6;
    registration.rejector.max_dist_sq = this->params_->icp.max_correspondence_distance * this->params_->icp.max_correspondence_distance;
    registration.criteria.rotation_eps = 0.1 * M_PI / 180.0;
    registration.criteria.translation_eps = 1e-3;
    registration.optimizer.max_iterations = 15;
    registration.optimizer.verbose = true;

    // This has been checked to be the correct transform (it's not the inverse of this)
    Isometry3d init_transform = T_corr_curr.inverse() * T_corr_prev;

    ROS_INFO("Running ICP.");
    ros::Time icp_start_time = ros::Time::now();

    // TO DO: make this work
    auto result = registration.align(*voxelgrid_prev, voxelgrid_accessor_curr, *voxelgrid_prev, init_transform); // voxelgrid accessor
    // auto result = registration.align(*voxelgrid_prev, *pointcloud_curr, *voxelgrid_prev, init_transform); // this one works
    // auto result = registration.align(*pointcloud_prev, *pointcloud_curr, init_transform);

    float fitness_score = result.error / result.num_inliers;
    Isometry3d correction_transform = result.T_target_source;
    ROS_INFO("ICP took %.3g ms. Fitness score: %.4g, converged: %d. Took %ld iterations. Num inliers: %ld", (ros::Time::now() - icp_start_time).toSec()*1e3, fitness_score, result.converged, result.iterations, result.num_inliers);

    // TO DO: Re-enable this
    // If fitness score is low but result didn't converge, search further
    // if (fitness_score < this->params_->icp.fitness_threshold*5 && !result.converged)
    // {
    //     registration.optimizer.max_iterations = 35;
    //     ROS_INFO("[Loop Closure] ICP had low fitness but didn't converge. Trying again with more iterations.");
    //     ros::Time icp_start_time2 = ros::Time::now();
    //     result = registration.align(*voxelgrid_prev, voxelgrid_accessor_curr, *voxelgrid_prev, result.T_target_source);
    //     ROS_INFO("Second ICP took %.3g ms. Fitness score: %.4g, converged: %d. Took %ld iterations.", (ros::Time::now() - icp_start_time2).toSec()*1e3, fitness_score, result.converged, result.iterations);

    //     // Update fitness score and transform
    //     fitness_score = result.error / result.num_inliers;
    //     correction_transform = result.T_target_source;
    // }

    // Error out if didn't converge
    if (!result.converged || std::isnan(fitness_score) || fitness_score > this->params_->icp.fitness_threshold) {
        ROS_WARN("[Loop Closure] ICP fitness test failed (%.4g > %.4g). Not adding loop closure between %d and %d. ICP runtime: %.3g ms. Distance: %.4g m.", fitness_score, this->params_->icp.fitness_threshold, lc_id_prev, lc_id_curr, (ros::Time::now() - icp_start_time).toSec()*1e3, correction_transform.translation().norm());
        return std::nullopt;
    }

    // Converged well, return answer and echo
    ROS_INFO("[Loop Closure] ICP fitness test passed (%.4g < %.4g). Adding loop closure between %d and %d. ICP runtime: %.3g ms. Distance: %.4g m.", fitness_score, this->params_->icp.fitness_threshold, lc_id_prev, lc_id_curr, (ros::Time::now() - icp_start_time).toSec()*1e3, correction_transform.translation().norm());

    return std::make_pair(correction_transform, fitness_score);

} // icp

/**
 * @brief Returns the list of added loop closures.
 */
std::vector<std::pair<int, int>> LoopClosureManager::getAddedLoopClosures() const
{
    std::lock_guard<std::mutex> lock(this->candidate_mutex_);
    return this->added_loop_closures_;
}

/**
 * @brief Finds the nearest loop closure keyframe to a given keyframe index.
 * @param graph_kf_index The index of the keyframe in the graph.
 * @return The lc index of the nearest keyframe.
 */
int LoopClosureManager::findNearestLCKeyframe(int target_graph_kf_index) const
{
    int best = -1;
    int min_diff = std::numeric_limits<int>::max();

    std::lock_guard<std::mutex> lock(this->kf_mutex_);
    for (size_t i=0; i<this->kf_indicies_.size(); ++i) {
        int graph_id = this->kf_indicies_.at(i);
        int diff = std::abs(graph_id - target_graph_kf_index);
        if (diff < min_diff) {
            min_diff = diff;
            best = i;
        }
    }
    return best;
}

} // namespace spade_pgo