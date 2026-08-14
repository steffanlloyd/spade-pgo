#include <queue>
#include <vector>
#include <mutex>
#include <optional>
#include <utility>
#include <limits>

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
    std::lock_guard<std::mutex> lock(this->candidate_mutex_);

    // Check if pair already tested.
    for (const auto& p : this->tested_candidates_) {
        if ((p.first == id1 && p.second == id2) ||
            (p.first == id2 && p.second == id1)) {
            ROS_DEBUG("Already detected loop between %d and %d", id1, id2);
            return;
        }
    }
    // Add candidate to queue and tested list
    this->candidate_queue_.emplace(id1, id2);
    this->tested_candidates_.emplace_back(id1, id2);
}

void LoopClosureManager::updateCandidateQueue()
{
    if(this->params_->near_kf.enabled){
        // Get and add near KF candidates
        auto candidates_near_kf = this->near_kf_detector.getNearKFCandidates(
            this->graph_manager_->getUpdatedKFPoses(),
            this->graph_manager_->getKeyframeSessionIds());

        for (const auto& candidate : candidates_near_kf){
            ROS_INFO("Added near_kf loop closure candidate between %d and %d", candidate.first, candidate.second);
            this->submitCandidate(candidate.first, candidate.second);
        }
    }

    // Only do SC if we have enough keyframes (and enabled)
    if( this->params_->sc.enabled &&
        this->graph_manager_->graphSize() > this->sc_detector.NUM_EXCLUDE_RECENT)
    {
        auto sc_candidate = this->sc_detector.detectLoopClosureID();
        if (sc_candidate.first != -1) {
            ROS_INFO("Added SC loop closure candidate between %d and %d", sc_candidate.first, this->graph_manager_->currentKFIndex());
            this->submitCandidate(sc_candidate.first, this->graph_manager_->currentKFIndex());
        }
    }
}

void LoopClosureManager::processCandidateQueue()
{
    bool queue_was_not_empty = this->getCandidateQueueSize() > 0;

    while (this->getCandidateQueueSize() > 0)
    {
        size_t queue_size = this->getCandidateQueueSize();
        if (queue_size > 200) {
            ROS_WARN_THROTTLE(2, "%ld loop closure candidates waiting... Adjust settings to produce fewer loop closure candidates", queue_size);
        }

        // Pop the queue with a mutex for safety
        std::pair<int, int> candidate_pair;
        {
            std::lock_guard<std::mutex> lock(this->candidate_mutex_);

            if (this->candidate_queue_.empty()) {
                break;
            }

            candidate_pair = this->candidate_queue_.front();

            // We can't process loop closures until the pose has been updated by the graph.
            // Skip any that are not updated yet.
            if(candidate_pair.first > this->graph_manager_->updatedKFIndex() || candidate_pair.second > this->graph_manager_->updatedKFIndex()){
                break;
            }

            this->candidate_queue_.pop();
        }

        // get BetweenFactor between poses (or std::nullopt if doesn't pass test)
        const int kf_prev = candidate_pair.first;
        const int kf_curr = candidate_pair.second;

        // Do ICP, add graph factor if valid (ICP is expensive, do outside mutex)
        if(auto relative_pose = this->icp(kf_prev, kf_curr)) {
            auto [T_between, fitness_score] = relative_pose.value();

            // Add to list of loop closures to add to graph
            this->graph_manager_->addLoopClosureFactor(kf_prev, kf_curr, T_between, fitness_score);

            // Add to list of added loop closures (thread-safe)
            {
                std::lock_guard<std::mutex> lock(this->candidate_mutex_);
                this->added_loop_closures_.emplace_back(kf_prev, kf_curr);
            }
        }

        ROS_INFO("Loop closure queue size: %ld", this->getCandidateQueueSize());
    }

    if (queue_was_not_empty && this->getCandidateQueueSize() == 0) {
        ROS_INFO("Finished processing loop closure candidates");
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

    // Preprocess clouds to remove canopy and distant points
    auto kf_poses = this->graph_manager_->getUpdatedKFPoses();
    this->preprocessPointCloud_(cloud_curr, kf_poses.at(kf_curr));
    this->preprocessPointCloud_(cloud_prev, kf_poses.at(kf_prev));

    // Check for insufficient points after preprocessing
    constexpr size_t MIN_POINTS_FOR_ICP = 100;
    if (cloud_curr->size() < MIN_POINTS_FOR_ICP || cloud_prev->size() < MIN_POINTS_FOR_ICP) {
        ROS_WARN("[Loop Closure] Insufficient points for ICP between %d and %d (curr: %zu, prev: %zu). Skipping.",
                 kf_prev, kf_curr, cloud_curr->size(), cloud_prev->size());
        return std::nullopt;
    }

    // Publish clouds, if in settings
    this->visualizer_->publishLCClouds(cloud_curr, cloud_prev);

    // Get the current time before ICP computation
    ros::Time icp_start_time = ros::Time::now();
    pcl::PointCloud<PointType>::Ptr reg_result(new pcl::PointCloud<PointType>());

    // Perform ICP
    auto source = geometry::pclToEigen(cloud_curr);
    auto target = geometry::pclToEigen(cloud_prev);
    small_gicp::RegistrationSetting settings;
    settings.num_threads = 12;                    // Number of threads to be used
    settings.max_correspondence_distance = this->params_->icp.max_correspondence_distance;  // Maximum correspondence distance between points (e.g., triming threshold)
    settings.voxel_resolution = 1.0;
    settings.max_iterations = this->params_->icp.max_iterations;
    settings.downsampling_resolution = this->params_->icp.voxel_size;

    if (this->params_->icp.algorithm == "small_gicp") settings.type = small_gicp::RegistrationSetting::RegistrationType::GICP;
    else if(this->params_->icp.algorithm == "small_vgicp") settings.type = small_gicp::RegistrationSetting::RegistrationType::VGICP;
    else ROS_ERROR("Invalid small_gicp method: %s", this->params_->icp.algorithm.c_str());

    Isometry3d init_transform = Isometry3d::Identity();
    if (this->params_->icp.zero_init_z) {
        // Vertical offset between two loop-closure keyframes is drift, not real: GNSS
        // altitude is downweighted by gps_noise_z_scale, so nothing else constrains z.
        // init_T is T_target_source, so it carries source (curr) into target (prev).
        init_transform.translation().z() =
            kf_poses.at(kf_prev).translation().z() - kf_poses.at(kf_curr).translation().z();
    }
    const double init_z = init_transform.translation().z();
    small_gicp::RegistrationResult result = small_gicp::align(target->points, source->points, init_transform, settings);

    Isometry3d correction_transform = result.T_target_source;
    double icp_runtime_ms = (ros::Time::now() - icp_start_time).toSec() * 1e3;

    // Guard against division by zero when no inliers found
    if (result.num_inliers == 0) {
        ROS_WARN("[Loop Closure] ICP found no inliers between %d and %d. Skipping. ICP runtime: %.3g ms. Init z: %.3g m.",
                 kf_prev, kf_curr, icp_runtime_ms, init_z);
        return std::nullopt;
    }

    // Check inlier ratio (use smaller cloud as reference for expected correspondences)
    const size_t min_cloud_size = std::min(source->size(), target->size());
    const float inlier_ratio = static_cast<float>(result.num_inliers) / static_cast<float>(min_cloud_size);
    if (this->params_->icp.min_inlier_ratio > 0 && inlier_ratio < this->params_->icp.min_inlier_ratio) {
        ROS_WARN("[Loop Closure] Low inlier ratio (%.1f%% < %.1f%%). Rejecting loop closure between %d and %d. ICP runtime: %.3g ms.",
                 inlier_ratio * 100, this->params_->icp.min_inlier_ratio * 100, kf_prev, kf_curr, icp_runtime_ms);
        return std::nullopt;
    }

    float fitness_score = result.error / result.num_inliers;

    if (!result.converged || std::isnan(fitness_score) || fitness_score > this->params_->icp.fitness_threshold) {
        ROS_WARN("[Loop Closure] ICP fitness test failed (%.4g > %.4g). Not adding loop closure between %d and %d. ICP runtime: %.3g ms. Distance: %.4g m. Inliers: %zu. Init z: %.3g m. Corr z: %.3g m.",
                 fitness_score, this->params_->icp.fitness_threshold, kf_prev, kf_curr, icp_runtime_ms, correction_transform.translation().norm(), result.num_inliers, init_z, correction_transform.translation().z());
        return std::nullopt;
    } else {
        ROS_INFO("[Loop Closure] ICP fitness test passed (%.4g < %.4g). Adding loop closure between %d and %d. ICP runtime: %.3g ms. Distance: %.4g m. Inliers: %zu. Init z: %.3g m. Corr z: %.3g m.",
                 fitness_score, this->params_->icp.fitness_threshold, kf_prev, kf_curr, icp_runtime_ms, correction_transform.translation().norm(), result.num_inliers, init_z, correction_transform.translation().z());
    }

    // Compose here, not in addLoopClosureFactor: correction_transform is only meaningful
    // against the kf_poses snapshot taken above, and iSAM2 may move them before the factor
    // is created. Result is the BetweenFactor measurement X_prev^-1 * X_curr_corrected.
    Eigen::Isometry3d T_between = kf_poses.at(kf_prev).inverse() * correction_transform * kf_poses.at(kf_curr);

    return std::make_pair(T_between, fitness_score);

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
 * @brief Returns the current size of the loop closure candidate queue.
 */
size_t LoopClosureManager::getCandidateQueueSize() const
{
    std::lock_guard<std::mutex> lock(this->candidate_mutex_);
    return this->candidate_queue_.size();
}

/**
 * @brief Returns the number of tested loop closure candidates.
 */
size_t LoopClosureManager::getTestedCandidatesCount() const
{
    std::lock_guard<std::mutex> lock(this->candidate_mutex_);
    return this->tested_candidates_.size();
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
        auto pointcloud = this->graph_manager_->getKeyframePointCloud(i);
        if (pointcloud) {
            *cloud_in += *geometry::pclTransform(pointcloud, kf_updated.at(i).matrix());
        }
    }

} // assembleNearbyKeyframesCloud_

/**
 * @brief Preprocesses a point cloud by filtering out high-altitude and distant points.
 *
 * For forest environments, the canopy makes ICP matching difficult. This function
 * removes points that are above a configurable height threshold relative to the
 * 2nd percentile z value (robust ground estimate), and points beyond a maximum
 * horizontal distance from the keyframe origin.
 *
 * @param cloud The point cloud to preprocess (modified in place).
 * @param T_origin The pose of the keyframe used as the origin for distance filtering.
 */
void LoopClosureManager::preprocessPointCloud_(pcl::PointCloud<PointType>::Ptr& cloud, const Eigen::Isometry3d& T_origin) const
{
    if (cloud->empty()) {
        return;
    }

    const bool filter_height = this->params_->icp.max_height_above_ground > 0;
    const bool filter_radius = this->params_->icp.max_radius_from_keyframe > 0;

    if (!filter_height && !filter_radius) {
        return;
    }

    // Calculate ground z using 2nd percentile (robust to outliers)
    float ground_z = 0.0f;
    if (filter_height) {
        std::vector<float> z_values;
        z_values.reserve(cloud->size());
        for (const auto& point : cloud->points) {
            z_values.push_back(point.z);
        }
        const size_t percentile_idx = static_cast<size_t>(z_values.size() * 0.02);
        std::nth_element(z_values.begin(), z_values.begin() + percentile_idx, z_values.end());
        ground_z = z_values[percentile_idx];
    }

    const float max_z = ground_z + static_cast<float>(this->params_->icp.max_height_above_ground);
    const float max_radius_sq = static_cast<float>(this->params_->icp.max_radius_from_keyframe *
                                                    this->params_->icp.max_radius_from_keyframe);
    const Eigen::Vector3f origin = T_origin.translation().cast<float>();

    pcl::PointCloud<PointType>::Ptr filtered_cloud(new pcl::PointCloud<PointType>());
    filtered_cloud->reserve(cloud->size());

    for (const auto& point : cloud->points) {
        // Height filter
        if (filter_height && point.z > max_z) continue;

        // Radius filter
        if (filter_radius) {
            const float dx = point.x - origin.x();
            const float dy = point.y - origin.y();
            if (dx * dx + dy * dy > max_radius_sq) continue;
        }

        filtered_cloud->push_back(point);
    }

    cloud->swap(*filtered_cloud);
}

} // namespace spade_pgo