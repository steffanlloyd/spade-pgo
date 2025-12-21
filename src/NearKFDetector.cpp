
#include <optional>
#include <vector>

#include "spade_pgo/common.hpp"
#include "spade_pgo/PGOParams.hpp"
#include "spade_pgo/NearKFDetector.hpp"
#include "spade_pgo/geometry.hpp"

#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <ros/ros.h>

namespace spade_pgo {

NearKFDetector::NearKFDetector(std::shared_ptr<PGOParams> params)
    :   params_(params),
        process_kf_id_(0),
        last_detected_lc_curr_(-1){}

NearKFDetector::~NearKFDetector() {}

/**
 * @brief Searches through keyframe poses for nearby loop closures based on distance.
 * @param kf_updated The updated keyframe poses.
 * @param kf_session_ids The session ID for each keyframe (for multi-session support).
 * @return A vector of pairs of indices representing the detected loop closure candidates.
 */
std::vector<std::pair<int, int>> NearKFDetector::getNearKFCandidates(
    const std::vector<Eigen::Isometry3d>& kf_updated,
    const std::vector<uint8_t>& kf_session_ids)
{
    std::vector<std::pair<int, int>> candidates;

    // Helper to get session ID safely
    auto getSessionId = [&](int idx) -> uint8_t {
        return (idx >= 0 && idx < static_cast<int>(kf_session_ids.size())) ? kf_session_ids[idx] : 0;
    };

    while (this->process_kf_id_ < static_cast<int>(kf_updated.size()) - 1) {
        // Increment the frame index
        this->process_kf_id_++;

        uint8_t curr_session = getSessionId(this->process_kf_id_);

        // Check if we are too close to a previously detected loop closure candidate (within same session only).
        if (this->last_detected_lc_curr_ >= 0 &&
            getSessionId(this->last_detected_lc_curr_) == curr_session &&
            geometry::euclideanDistance2(kf_updated.at(this->process_kf_id_), kf_updated.at(this->last_detected_lc_curr_)) <
            std::pow(this->params_->near_kf.min_consecutive_kf_distance, 2))
        {
            continue;
        }

        // Iterate over all previous frames to detect nearby keyframes.
        int last_detected_lc_prev = -1;
        for (int j = 0; j < static_cast<int>(kf_updated.size()); ++j) {
            uint8_t j_session = getSessionId(j);
            bool same_session = (j_session == curr_session);

            // Only apply min_kf_seperation for keyframes within the same session
            if (same_session && (this->process_kf_id_ - j < this->params_->near_kf.min_kf_seperation)) {
                continue;
            }

            // Skip if too close to previously detected loop closure in previous frames (within same session only)
            if (last_detected_lc_prev >= 0 &&
                getSessionId(last_detected_lc_prev) == j_session &&
                geometry::euclideanDistance2(kf_updated.at(j), kf_updated.at(last_detected_lc_prev)) <
                std::pow(this->params_->near_kf.min_consecutive_kf_distance, 2))
            {
                continue;
            }

            double distance2 = geometry::euclideanDistance2(kf_updated.at(j), kf_updated.at(this->process_kf_id_));

            // If the distance between keyframes is small, add it as a candidate
            if (distance2 < std::pow(this->params_->near_kf.distance_threshold, 2)) {
                candidates.emplace_back(j, this->process_kf_id_);
                this->last_detected_lc_curr_ = this->process_kf_id_;
                last_detected_lc_prev = j;

                if (same_session) {
                    ROS_INFO("Intra-session LC candidate: %d <-> %d (%.2fm)", j, this->process_kf_id_, std::sqrt(distance2));
                } else {
                    ROS_INFO("Inter-session LC candidate: %d (s%d) <-> %d (s%d) (%.2fm)",
                             j, j_session, this->process_kf_id_, curr_session, std::sqrt(distance2));
                }
            }
        }
    }

    return candidates;
}


} // namespace spade_pgo