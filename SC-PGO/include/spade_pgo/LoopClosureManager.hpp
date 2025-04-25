#pragma once

#include <queue>
#include <vector>
#include <mutex>
#include <optional>
#include <utility>

#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "spade_pgo/PoseGraphManager.hpp"
#include "spade_pgo/common.hpp"
#include "spade_pgo/Visualizer.hpp"
#include "spade_pgo/PGOParams.hpp"
#include "spade_pgo/NearKFDetector.hpp"

#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/incremental_voxelmap.hpp>

#include "scancontext/Scancontext.h"

namespace spade_pgo {

class LoopClosureManager {
public:
    LoopClosureManager(
        std::shared_ptr<PoseGraphManager> graph_manager,
        std::shared_ptr<Visualizer> visualizer,
        std::shared_ptr<PGOParams> params);
    ~LoopClosureManager();

    void submitCandidate(int id1, int id2);
    void updateCandidateQueue();
    void processCandidateQueue();
    void submitKeyframeCloud(int kf_index, const Eigen::Isometry3d &T, pcl::PointCloud<PointType>::Ptr &pointcloud);
    std::optional<std::pair<Eigen::Isometry3d, double>> icp( int kf_prev, int kf_curr );

    std::vector<std::pair<int, int>> getAddedLoopClosures() const;

    int findNearestLCKeyframe(int target_graph_kf_index) const;

    SCManager sc_detector;
    NearKFDetector near_kf_detector;

private:
    void assembleNearbyKeyframesCloud_(
        pcl::PointCloud<PointType>::Ptr& cloud_in,
        int kf_index,
        int num_kf_accumulate) const;

    std::shared_ptr<PoseGraphManager> graph_manager_;
    std::shared_ptr<Visualizer> visualizer_;
    std::shared_ptr<PGOParams> params_;

    std::vector<int> kf_indicies_;
    std::vector<std::shared_ptr<small_gicp::IncrementalVoxelMap<small_gicp::FlatContainerNormalCov>>> kf_pointclouds_;

    std::queue<std::pair<int, int>> candidate_queue_;
    std::vector<std::pair<int, int>> tested_candidates_;
    std::vector<std::pair<int, int>> added_loop_closures_;

    pcl::VoxelGrid<PointType> voxelizer_sc_;

    mutable std::mutex candidate_mutex_;
    mutable std::mutex kf_mutex_;
};

} // namespace spade_pgo