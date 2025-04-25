
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
 * @return A vector of pairs of indices representing the detected loop closure candidates.
 */
std::vector<std::pair<int, int>> NearKFDetector::getNearKFCandidates(const std::vector<int> lc_indicies, const std::vector<Eigen::Isometry3d>& kf_updated)
{
    std::vector<std::pair<int, int>> candidates;

    while (this->process_kf_id_ < static_cast<int>(lc_indicies.size()) - 1) {

        // Check if the next index is in the updated list, otherwise break and do it later
        if( lc_indicies.at(this->process_kf_id_+1) >= static_cast<int>(kf_updated.size()) )
        {
            break;
        }

        // Increment the frame index
        this->process_kf_id_++;
        auto T_curr = kf_updated.at(lc_indicies.at(this->process_kf_id_));


        // Check if we are too close to a previously detected loop closure candidate.
        if (this->last_detected_lc_curr_ >= 0){

            auto T_last_lc_curr = kf_updated.at(lc_indicies.at(this->last_detected_lc_curr_));

            if(geometry::euclideanDistance2(T_curr, T_last_lc_curr) < 
            std::pow(this->params_->near_kf.min_consecutive_kf_distance, 2)){
                continue;
            }
        }

        // Iterate over previous frames to detect nearby keyframes.
        int last_detected_lc_prev = -1;
        for (int j = 0; j < static_cast<int>(lc_indicies.size()) - this->params_->near_kf.min_kf_seperation; ++j) {

            auto T_j = kf_updated.at(lc_indicies.at(j));

            // Check if the index isn't too close to a previously detected loop closure
            if (last_detected_lc_prev >= 0){
                auto T_last_lc_prev = kf_updated.at(this->process_kf_id_);
                if (geometry::euclideanDistance2(T_j, T_last_lc_prev) < 
                    std::pow(this->params_->near_kf.min_consecutive_kf_distance, 2))
                {
                    continue;
                }
            }

            double distance2 = geometry::euclideanDistance2(T_j, T_curr);

            // If the distance between keyframes is small, add it as a candidate
            if (distance2 < std::pow(this->params_->near_kf.distance_threshold, 2)) {
                candidates.emplace_back( lc_indicies.at(j), lc_indicies.at(this->process_kf_id_));
                this->last_detected_lc_curr_ = this->process_kf_id_;
                last_detected_lc_prev = j;
                ROS_INFO("Added candidate loop closure pair %d and %d, distance %.3f m < %.3f m", lc_indicies.at(j), lc_indicies.at(this->process_kf_id_), std::sqrt(distance2), this->params_->near_kf.distance_threshold);
            }
        }
    }

    return candidates;
}


} // namespace spade_pgo