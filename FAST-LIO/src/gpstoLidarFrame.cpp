#include <ros/ros.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

using namespace std;

class GPSToLIOTransformer
{
public:
    GPSToLIOTransformer()
    {
        // Initialize flags
        initial_transform_computed_ = false;

        // Subscribe to GPS data
        gps_sub_ = nh_.subscribe("/mavros/global_position/local", 1, &GPSToLIOTransformer::gpsCallback, this);

        // Subscribe to LIO odometry data
        lio_sub_ = nh_.subscribe("/Odometry", 1, &GPSToLIOTransformer::lioCallback, this);

        // Publisher for transformed GPS data
        transformed_gps_pub_ = nh_.advertise<nav_msgs::Odometry>("/gps/odometry_transformed", 1);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber gps_sub_;
    ros::Subscriber lio_sub_;
    ros::Publisher transformed_gps_pub_;

    // Store initial poses
    geometry_msgs::Pose initial_gps_pose_;
    geometry_msgs::Pose initial_lio_pose_;

    // Transformation between GPS and LIO frames
    tf::Transform gps_to_lio_transform_;

    // Flag to indicate if the initial transform has been computed
    bool initial_transform_computed_;

    void gpsCallback(const nav_msgs::Odometry::ConstPtr& gps_msg)
    {
        // Store the initial GPS pose if not already done
        if (!initial_transform_computed_)
        {
            initial_gps_pose_ = gps_msg->pose.pose;
            computeInitialTransform();
        }

        // Apply the transformation to GPS data
        if (initial_transform_computed_)
        {
            // Convert GPS pose to tf format
            tf::Pose gps_pose_tf;
            tf::poseMsgToTF(gps_msg->pose.pose, gps_pose_tf);
            // Set the Z component of the position to zero
            gps_pose_tf.getOrigin().setZ(0.0);

            // Transform the GPS pose to LIO frame
            tf::Pose transformed_pose_tf = gps_to_lio_transform_ * gps_pose_tf;

            // Convert back to Pose message
            geometry_msgs::Pose transformed_pose_msg;
            tf::poseTFToMsg(transformed_pose_tf, transformed_pose_msg);

            // Publish as Odometry message
            nav_msgs::Odometry transformed_odom;
            transformed_odom.header.stamp = gps_msg->header.stamp;
            transformed_odom.header.frame_id = "camera_init";  // Replace with your LIO frame ID
            transformed_odom.child_frame_id = "gps_transformed";
            transformed_odom.pose.pose = transformed_pose_msg;
            transformed_odom.pose.covariance = gps_msg->pose.covariance;

            transformed_gps_pub_.publish(transformed_odom);
        }
    }

    void lioCallback(const nav_msgs::Odometry::ConstPtr& lio_msg)
    {
        // Store the initial LIO pose if not already done
        if (!initial_transform_computed_)
        {
            initial_lio_pose_ = lio_msg->pose.pose;
            computeInitialTransform();
        }
    }

    void computeInitialTransform()
    {
        // Check if both initial poses have been received
        if (initial_gps_pose_.position.x != 0 || initial_gps_pose_.position.y != 0 || initial_gps_pose_.position.z != 0)
        {
            if (initial_lio_pose_.position.x != 0 || initial_lio_pose_.position.y != 0 || initial_lio_pose_.position.z != 0)
            {
                // Convert poses to tf format
                tf::Pose gps_pose_tf, lio_pose_tf;
                initial_gps_pose_.position.z = 0;

                // Set roll and pitch of initial_gps_pose_ to zero
                tf::Quaternion gps_orientation;
                tf::quaternionMsgToTF(initial_gps_pose_.orientation, gps_orientation);

                double roll, pitch, yaw;
                tf::Matrix3x3(gps_orientation).getRPY(roll, pitch, yaw);

                roll = 0.0;
                pitch = 0.0;

                gps_orientation.setRPY(roll, pitch, yaw);
                gps_orientation.normalize();

                tf::quaternionTFToMsg(gps_orientation, initial_gps_pose_.orientation);

                tf::poseMsgToTF(initial_gps_pose_, gps_pose_tf);
                tf::poseMsgToTF(initial_lio_pose_, lio_pose_tf);

                // Compute the transform from GPS to LIO frame
                gps_to_lio_transform_ = lio_pose_tf * gps_pose_tf.inverse();

                // Instead of computing the transform, set it directly
                tf::Vector3 translation(-0.699747, 3.020508, -0.120096);
                tf::Quaternion rotation(-0.051851, -0.069255, 0.607455, 0.789629);
                rotation.normalize();  // Ensure the quaternion is normalized

                gps_to_lio_transform_.setOrigin(translation);
                gps_to_lio_transform_.setRotation(rotation);
                
                initial_transform_computed_ = true;

                ROS_INFO("Initial transform between GPS and LIO frames computed.");
                printTransform(gps_to_lio_transform_);
            }
        }
    }

    // Function to print the transform
    void printTransform(const tf::Transform& transform)
    {
        tf::Vector3 translation = transform.getOrigin();
        tf::Quaternion rotation = transform.getRotation();

        double roll, pitch, yaw;
        tf::Matrix3x3(rotation).getRPY(roll, pitch, yaw);

        ROS_INFO("Transformation from GPS to LIO Frame:");
        ROS_INFO("Translation: x = %f, y = %f, z = %f", translation.x(), translation.y(), translation.z());
        ROS_INFO("Rotation (quaternion): x = %f, y = %f, z = %f, w = %f", rotation.x(), rotation.y(), rotation.z(), rotation.w());
        ROS_INFO("Rotation (Euler angles): roll = %f, pitch = %f, yaw = %f (radians)", roll, pitch, yaw);
        ROS_INFO("Rotation (Euler angles): roll = %f, pitch = %f, yaw = %f (degrees)", roll * 180.0 / M_PI, pitch * 180.0 / M_PI, yaw * 180.0 / M_PI);
    }

};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "gps_to_lio_frame_transformer");
    GPSToLIOTransformer transformer;
    ros::spin();
    return 0;
}
