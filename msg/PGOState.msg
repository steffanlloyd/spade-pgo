# PGO State message for monitoring multi-drone swarm processing
std_msgs/Header header

# Current session info
uint8 current_drone_id
uint32 num_keyframes
uint32 num_keyframes_current_session

# Loop closure queue status
uint32 lc_candidate_queue_size
uint32 lc_tested_count
uint32 lc_added_count

# Graph status
bool graph_initialized
bool gnss_initialized

# Session boundaries (for debugging/visualization)
uint32[] session_start_indices    # Keyframe indices where each session starts
uint8[] session_drone_ids         # Drone ID for each session
