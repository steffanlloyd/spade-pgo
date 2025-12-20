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
    void getNearKFCandidates();
    std::optional<std::pair<Eigen::Isometry3d, double>> icp( int kf_prev, int kf_curr );

    std::vector<std::pair<int, int>> getAddedLoopClosures() const;
    size_t getCandidateQueueSize() const;
    size_t getTestedCandidatesCount() const;

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

    std::queue<std::pair<int, int>> candidate_queue_;
    std::vector<std::pair<int, int>> tested_candidates_;
    std::vector<std::pair<int, int>> added_loop_closures_;

    mutable std::mutex candidate_mutex_;
};

} // namespace spade_pgo