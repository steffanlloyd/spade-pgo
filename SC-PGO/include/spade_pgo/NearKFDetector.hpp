#pragma once
#include <optional>
#include <vector>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "spade_pgo/common.hpp"
#include "spade_pgo/PGOParams.hpp"

namespace spade_pgo {

class NearKFDetector{
public:
    NearKFDetector(std::shared_ptr<PGOParams> params);
    ~NearKFDetector();

    std::vector<std::pair<int, int>> getNearKFCandidates(const std::vector<Eigen::Isometry3d>& kf_updated);

private:
    std::shared_ptr<PGOParams> params_;

    int process_kf_id_;
    int last_detected_lc_curr_;
};

} // namespace spade_pgo