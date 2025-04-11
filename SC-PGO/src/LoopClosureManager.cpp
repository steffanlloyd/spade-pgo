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
}
LoopClosureManager::~LoopClosureManager() {}

/**
 * @brief Submits a loop closure candidate if it hasn't been tested yet.
 */
void LoopClosureManager::submitCandidate(int id1, int id2)
{
    // Check if pair already tested.
    for (const auto& p : this->tested_candidates_) {
        if ((p.first == id1 && p.second == id2) ||
            (p.first == id2 && p.second == id1)) {
            ROS_DEBUG("Already detected loop between %d and %d", id1, id2);
            return;
        }
    }
    // Add candidate in a thread-safe manner.
    {
        std::lock_guard<std::mutex> lock(this->candidate_mutex_);
        this->candidate_queue_.emplace(id1, id2);
    }
    // Add to list of tested candidates
    this->tested_candidates_.emplace_back(id1, id2);
    ROS_INFO("Added loop closure candidate between %d and %d", id1, id2);
}

void LoopClosureManager::updateCandidateQueue()
{
    if(this->params_->loop_closure.use_near_kf){
        // Get and add near KF candidates
        auto candidates_near_kf = this->near_kf_detector.getNearKFCandidates(this->graph_manager_->getUpdatedKFPoses());

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
            if(candidate_pair.first >= this->graph_manager_->updatedKFIndex() || candidate_pair.second >= this->graph_manager_->updatedKFIndex()){
                break;
            }
            
            this->candidate_queue_.pop();
        }

        // get BetweenFactor between poses (or std::nullopt if doesn't pass test)
        const int kf_prev = candidate_pair.first;
        const int kf_curr = candidate_pair.second;
        
        // Do ICP, add graph factor if valid
        if(auto relative_pose = this->icp(kf_prev, kf_curr)) {
            auto [Ticp, fitness_score] = relative_pose.value();

            // Add to list of loop closures to add to graph
            this->graph_manager_->addLoopClosureFactor(kf_prev, kf_curr, Ticp, fitness_score);

            // Add to list of added loop closures
            this->added_loop_closures_.emplace_back(kf_prev, kf_curr);
        } 
    }
}

/**
 * @brief Performs ICP to find the transformation between two keyframes.
 * @param kf_prev The index of the previous keyframe.
 * @param kf_curr The index of the current keyframe.
 * @return A pair containing the transformation and fitness score, or nullopt if ICP fails.
 */
std::optional<std::pair<Eigen::Isometry3d, double>> LoopClosureManager::icp( int kf_prev, int kf_curr )
{
    // SL: Define point clouds
    pcl::PointCloud<PointType>::Ptr cloud_curr(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr cloud_prev(new pcl::PointCloud<PointType>());     

    // Assemble point clouds from frames around potential key frames indicies
    this->assembleNearbyKeyframesCloud_(cloud_curr, kf_curr, this->params_->icp.num_kf_accumulate_now);
    this->assembleNearbyKeyframesCloud_(cloud_prev, kf_prev, this->params_->icp.num_kf_accumulate_past);   

    // Publish clouds, if in settings
    this->visualizer_->publishLCClouds(cloud_curr, cloud_prev);
    
    // Get the current time before ICP computation
    ros::Time icp_start_time = ros::Time::now();
    pcl::PointCloud<PointType>::Ptr reg_result(new pcl::PointCloud<PointType>());

    // Perform ICP
    auto source = geometry::pclToEigen(cloud_curr);
    auto target = geometry::pclToEigen(cloud_prev); 
    small_gicp::RegistrationSetting settings;
    settings.num_threads = 6;                    // Number of threads to be used
    settings.max_correspondence_distance = this->params_->icp.max_correspondence_distance;  // Maximum correspondence distance between points (e.g., triming threshold)
    settings.voxel_resolution = 1.0;
    settings.max_iterations = 50;
    settings.downsampling_resolution = this->params_->icp.voxel_size;

    if (this->params_->icp.algorithm == "small_gicp") settings.type = small_gicp::RegistrationSetting::RegistrationType::GICP;
    else if(this->params_->icp.algorithm == "small_vgicp") settings.type = small_gicp::RegistrationSetting::RegistrationType::VGICP;
    else ROS_ERROR("Invalid small_gicp method: %s", this->params_->icp.algorithm.c_str());

    Isometry3d init_transform = Isometry3d::Identity();
    small_gicp::RegistrationResult result = small_gicp::align(target->points, source->points, init_transform, settings);

    float fitness_score = result.error / result.num_inliers;
    Isometry3d correction_transform = result.T_target_source;

    if (!result.converged || std::isnan(fitness_score) || fitness_score > this->params_->icp.fitness_threshold) {
        ROS_WARN("[Loop Closure] ICP fitness test failed (%.4g > %.4g). Not adding loop closure between %d and %d. ICP runtime: %.3g ms. Distance: %.4g m.", fitness_score, this->params_->icp.fitness_threshold, kf_prev, kf_curr, (ros::Time::now() - icp_start_time).toSec()*1e3, correction_transform.translation().norm());
        return std::nullopt;
    } else {
        ROS_INFO("[Loop Closure] ICP fitness test passed (%.4g < %.4g). Adding loop closure between %d and %d. ICP runtime: %.3g ms. Distance: %.4g m.", fitness_score, this->params_->icp.fitness_threshold, kf_prev, kf_curr, (ros::Time::now() - icp_start_time).toSec()*1e3, correction_transform.translation().norm());
    }

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
 * @brief Assembles nearby keyframes cloud for loop closure detection.
 * @param[in] cloud The point cloud to be assembled.
 * @param kf_index The index of the keyframe.
 * @param num_kf_accumulate The number of keyframes to accumulate around the desired keyframe
 */
void LoopClosureManager::assembleNearbyKeyframesCloud_(
    pcl::PointCloud<PointType>::Ptr& cloud_in,
    int kf_index,
    int num_kf_accumulate) const
{
    // extract and stacking near keyframes (in global coord)
    cloud_in->clear();
    auto kf_updated = this->graph_manager_->getUpdatedKFPoses();
    for (int i = kf_index-num_kf_accumulate; i <= kf_index+num_kf_accumulate; ++i) {
        if (i < 0 || i >= static_cast<int>(kf_updated.size())) continue;
        *cloud_in += *geometry::pclTransform(this->graph_manager_->kf_pointclouds.at(i), kf_updated.at(i).matrix());
    }

} // assembleNearbyKeyframesCloud_

} // namespace spade_pgo