#!/bin/python3

import gymnasium as gym
from gymnasium import utils
from gymnasium import spaces
from gymnasium.envs.registration import register
from frobs_rl.common import ros_gazebo
from frobs_rl.common import ros_controllers
from frobs_rl.common import ros_node
from frobs_rl.common import ros_launch
from frobs_rl.common import ros_params
from frobs_rl.common import ros_urdf
from frobs_rl.common import ros_spawn
from frobs_rl.envs import robot_BasicEnv
from open_manipulator_msgs.msg import KinematicsPose
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
from sensor_msgs.msg import Image, JointState, Imu, LaserScan
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import torch
# from robomaster_om_reacher.robot_env import robomaster_om_moveit
import rospy

import rostopic

import tf
import tf2_ros
from tf2_geometry_msgs import do_transform_point


from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Twist, PointStamped
from std_msgs.msg import Bool
from gazebo_msgs.srv import SetLinkState, SetLinkStateRequest, SetLinkStateResponse

import numpy as np
import scipy.spatial
import csv
import os
from datetime import datetime

register(
        id='Robomaster_OM_ReacherEnv-v0',
        entry_point='robomaster_om_reacher.task_env.robomaster_om_reacher:Robomaster_OM_ReacherEnv',
        max_episode_steps=10000000
    )

class Robomaster_OM_ReacherEnv(robot_BasicEnv.RobotBasicEnv):
    """
    Custom Task Env, use this env to implement a task using the robot defined in the CustomRobotEnv
    """

    def __init__(self):
        """
        Describe the task.
        """
        rospy.logwarn("Starting Robomaster_OM_ReacherEnv Task Env")

        # Setup CSV logging for rewards
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(os.path.dirname(__file__), '../logs/reward_logs')
        os.makedirs(log_dir, exist_ok=True)
        self.reward_log_file = os.path.join(log_dir, f'rewards_{timestamp}.csv')
        self.reward_fieldnames = ['timestep', 'total_reward', 'distance_penalty', 'prev_dist_reward', 'obstacle_avoidance_reward', 'goal_reward']
        
        with open(self.reward_log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.reward_fieldnames)
            writer.writeheader()

        # Setup CSV logging for observations
        obs_log_dir = os.path.join(os.path.dirname(__file__), '../logs/observation_logs')
        os.makedirs(obs_log_dir, exist_ok=True)
        self.obs_log_file = os.path.join(obs_log_dir, f'observations_{timestamp}.csv')
        self.obs_fieldnames = [
            'timestep',
            'vec_EE_GOAL_x', 'vec_EE_GOAL_y', 'vec_EE_GOAL_z',
            'current_goal_x', 'current_goal_y', 'current_goal_z',
            'ee_pose_x', 'ee_pose_y', 'ee_pose_z',
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
            'action_state_prev_robomaster_lin_x', 'action_state_prev_robomaster_lin_y', 'action_state_prev_robomaster_ang_z',
            'ee_to_goal_robomaster_lin_x', 'ee_to_goal_robomaster_lin_y', 'ee_to_goal_robomaster_ang_z'
        ]
        
        with open(self.obs_log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.obs_fieldnames)
            writer.writeheader()

        # Setup CSV logging for actions
        action_log_dir = os.path.join(os.path.dirname(__file__), '../logs/action_logs')
        os.makedirs(action_log_dir, exist_ok=True)
        self.action_log_file = os.path.join(action_log_dir, f'actions_{timestamp}.csv')
        self.action_fieldnames = [
            'timestep',
            'joint1_action', 'joint2_action', 'joint3_action', 'joint4_action', 'joint5_action', 'joint6_action',
            'robomaster_lin_x', 'robomaster_lin_y', 'robomaster_ang_z'
        ]
        
        with open(self.action_log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.action_fieldnames)
            writer.writeheader()
        
        self.timestep_counter = 0

        """
        Load YAML param file
        """
        ros_params.ros_load_yaml_from_pkg("robomaster_om_reacher", "reacher_task.yaml", ns="/") 
        self.get_params()

        """
        Init necessary variables and objects.
        """
        self.bridge = CvBridge()

        self.ee_pose = None
        self.joint_values = None
        self.goal_in_ee = None
        self.init_pos = np.array([0.0, -0.78, 1.5, 0.0, 0.8, 0.0])
        self.robomaster_action = None

        self.speed_scale = 1.0
        self.arrow_scale = 0.0
        self.prev_arrow_scale = 0.0

        self.action_state = None
        self.action_state_prev = None # to reduce the gap between the actions

        self.joint_state_topic = "/joint_states"

        # status variables of the robot
        self.stuck_open_manipulator = False
        self.is_flipped = False
        self.is_stuck = False

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(2.0))
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        self.ee_to_goal_robomaster_action = np.zeros(3, dtype=np.float32)
        self.min_distance = None


        # for debugging
        self.time_count = 0

        """
        Define the action and observation space.
        """
        
        #--- Define the ACTION SPACE
        # Define a continuous space using BOX and defining its limits
        # rospy.logwarn("dtype of min_joint_pos_values: " + str(type(self.min_joint_pos_values)))
        self.min_joint_pos_values = np.array(self.min_joint_pos_values, dtype=np.float32)
        self.max_joint_pos_values = np.array(self.max_joint_pos_values, dtype=np.float32)

        self.robomaster_min_vel_value_action = np.array([self.robomaster_min_vel_value["lin_x"], self.robomaster_min_vel_value["lin_y"], self.robomaster_min_vel_value["ang_z"]], dtype=np.float32)
        self.robomaster_max_vel_value_action = np.array([self.robomaster_max_vel_value["lin_x"], self.robomaster_max_vel_value["lin_y"], self.robomaster_max_vel_value["ang_z"]], dtype=np.float32)

        action_space_low = np.concatenate([self.min_joint_pos_values, self.robomaster_min_vel_value_action]) 
        action_space_high = np.concatenate([self.max_joint_pos_values, self.robomaster_max_vel_value_action])

        # normalized action space
        self.action_space = spaces.Box(low=action_space_low, high=action_space_high, shape=action_space_low.shape, dtype=np.float32)

        #--- Define the OBSERVATION SPACE

        #- manipulator joint position
        observations_om_joint_pos_max_range = np.array(self.max_joint_pos_values)
        observations_om_joint_pos_min_range = np.array(self.min_joint_pos_values)
        #- manipulator joint velocity   
        self.min_joint_vel_values = np.array(self.min_joint_vel_values, dtype=np.float32)
        self.max_joint_vel_values = np.array(self.max_joint_vel_values, dtype=np.float32)
        observations_om_joint_vel_max_range = np.array(self.max_joint_vel_values)
        observations_om_joint_vel_min_range = np.array(self.min_joint_vel_values)
        #- robomaster velocity
        # rospy.logwarn("dtype of robomaster_max_vel_value: " + str(type(self.robomaster_max_vel_value)))
        self.observations_robomaster_vel_max_range = np.array(np.array([self.robomaster_max_vel_value["lin_x"], self.robomaster_max_vel_value["lin_y"], self.robomaster_max_vel_value["ang_z"]]))
        self.observations_robomaster_vel_min_range = np.array(np.array([self.robomaster_min_vel_value["lin_x"], self.robomaster_min_vel_value["lin_y"], self.robomaster_min_vel_value["ang_z"]]))
        #- setpoint position of ee frame
        observations_high_setposition_pos_range = np.array(np.array([-5.0, -5.0, 0.1]))
        observations_low_setposition_pos_range  = np.array(np.array([5.0, 5.0, 0.45]))

        self.low_goal_pos_range_global = np.array([self.position_goal_min["x"], self.position_goal_min["y"], self.position_goal_min["z"]])
        self.high_goal_pos_range_global = np.array([self.position_goal_max["x"], self.position_goal_max["y"], self.position_goal_max["z"]])                
        

        #--- Observation space 
        self.observation_space = spaces.Dict({
            "scan_matrix": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(224, 224),  # LaserScan -> Tensor Obs shape for ViT
                # shape=(64, 64),  # LaserScan -> Tensor Obs shape for CNN
                dtype=np.float32
            ),
            "vector": spaces.Box(
                low=np.array([
                    -1.0, -1.0, -1.0,  # vec_EE_GOAL (normalized)
                    *self.low_goal_pos_range_global,  # current_goal
                    *self.low_goal_pos_range_global,  # ee_pose
                    *observations_om_joint_pos_min_range,  # joint_values
                    *self.observations_robomaster_vel_min_range,  # robomaster_vel
                    *self.observations_robomaster_vel_min_range  # robomaster_vel for ee_to_goal_robomaster_action
                ], dtype=np.float32),
                high=np.array([
                    1.0, 1.0, 1.0,  # vec_EE_GOAL (normalized)
                    *self.high_goal_pos_range_global,  # current_goal
                    *self.high_goal_pos_range_global,  # ee_pose
                    *observations_om_joint_pos_max_range,  # joint_values
                    *self.observations_robomaster_vel_max_range,  # robomaster_vel
                    *self.observations_robomaster_vel_max_range  # robomaster_vel for ee_to_goal_robomaster_action
                ], dtype=np.float32),
                shape=(21,),  # Combined size of vec_EE_GOAL (3) + current_goal (3) + ee_pose (3) + joint_values (6)
                dtype=np.float32
            )
        })
       
        #-- Action space for sampling
        self.goal_space = spaces.Box(low=self.low_goal_pos_range_global, high=self.high_goal_pos_range_global, dtype=np.float32)

        """
        Define subscribers or publishers as needed.
        """

        #--- Make Marker msg for publishing
        self.goal_marker = Marker()
        self.goal_marker.header.frame_id="odom"
        self.goal_marker.header.stamp = rospy.Time.now()
        self.goal_marker.ns = "goal_shapes"
        self.goal_marker.id = 0
        self.goal_marker.type = Marker.SPHERE
        self.goal_marker.action = Marker.ADD

        self.goal_marker.pose.position.x = 0.0
        self.goal_marker.pose.position.y = 0.0
        self.goal_marker.pose.position.z = 0.0
        self.goal_marker.pose.orientation.x = 0.0
        self.goal_marker.pose.orientation.y = 0.0
        self.goal_marker.pose.orientation.z = 0.0
        self.goal_marker.pose.orientation.w = 1.0

        self.goal_marker.scale.x = 0.3
        self.goal_marker.scale.y = 0.3
        self.goal_marker.scale.z = 0.3

        # color like tennis ball 
        self.goal_marker.color.r = 0.8
        self.goal_marker.color.g = 1.0
        self.goal_marker.color.b = 0.2
        self.goal_marker.color.a = 1.0

        self.pub_marker = rospy.Publisher("goal_point", Marker, queue_size=10)

        #--- Make Marker msg(arrow) for publishing
        self.arrow_marker = Marker()
        self.arrow_marker.header.frame_id="end_effector_link"
        self.arrow_marker.header.stamp = rospy.Time.now()
        self.arrow_marker.ns = "ee_goal_arrow"
        self.arrow_marker.id = 1
        self.arrow_marker.type = Marker.ARROW
        self.arrow_marker.action = Marker.ADD

        self.arrow_marker.color.r = 0.0
        self.arrow_marker.color.g = 0.0
        self.arrow_marker.color.b = 1.0
        self.arrow_marker.color.a = 0.8

        self.pub_arrow = rospy.Publisher("arrow_point", Marker, queue_size=10)

        self.robomaster_action_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.is_robomaster_flipped_pub = rospy.Publisher('/is_flip', Bool, queue_size=10)
        self.is_robomaster_stuck_pub = rospy.Publisher('/is_stuck', Bool, queue_size=10)

        # self.goal_subs  = rospy.Subscriber("/goal_pos", Point, self.goal_callback)
        self.scan_images_subs = rospy.Subscriber("/scan_image", Image, self.scan_image_callback)
        self.scan_dist_check_subs = rospy.Subscriber("/min_distance_threshold_breached", Bool, self.scan_dist_check_callback)
        # self.ee_pose_subs = rospy.Subscriber("/gripper/kinematics_pose", KinematicsPose, self.ee_pose_callback)
        self.joint_states_subs = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)
        self.imu_callback_subs = rospy.Subscriber("/imu/data", Imu, self.imu_callback)
        # self.odom_callback_subs = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.lidar_callback_subs = rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        self.scan_images_tensor = None
        self.scan_dist_check_breached = None
            
        # if self.training:
            # ros_node.ros_node_from_pkg("robomaster_om_reacher", "pos_publisher.py", name="pos_publisher", ns="/")
            # rospy.wait_for_service("set_init_point")
            # self.set_init_goal_client = rospy.ServiceProxy("set_init_point", SetLinkState)

        # self.model_name_in_gazebo = "robot"
        # self.namespace = "/"
        pkg_name = "robot_description"
        urdf_file = "robot_description.urdf.xacro" 

        """
        Init super class.
        """
        self.model_pos_x = 0.0
        self.model_pos_y = 0.3
        self.model_pos_z = 0.247
        
        # self.world_path = "/root/catkin_ws/src/aws-robomaker-small-warehouse-world/worlds/no_roof_small_warehouse.world"
        reset_mode = 1
        step_mode = 1
        
        self.use_sim_time = False
        self.model_name_in_gazebo = "robot"
        rospy.logwarn("Init super class")
        super(Robomaster_OM_ReacherEnv, self).__init__( launch_gazebo=False, spawn_robot=True, gazebo_freq = 10, model_pos_x=self.model_pos_x, model_pos_y=self.model_pos_y, model_pos_z=self.model_pos_z, pkg_name=pkg_name, urdf_file=urdf_file, urdf_folder="/robot",
                    model_name_in_gazebo=self.model_name_in_gazebo, reset_mode=reset_mode, step_mode=step_mode, use_sim_time = self.use_sim_time)
        
        ros_launch.ros_launch_from_pkg("robot_control", "robot_control.launch")  
        """
        Finished __init__ method
        """
        rospy.logwarn("Finished Init of Robomaster_OM_ReacherEnv Task Env")
        self.last_reward_time = rospy.get_time()
        self.last_update_goal_time = rospy.get_time()
        self.last_get_ee_pose_time = rospy.get_time()
        self.last_manipulator_action_time = rospy.get_time()

    #-------------------------------------------------------#
    #   Custom available methods for the CustomTaskEnv      #

    def _check_subs_and_pubs_connection(self):
        """
        Function to check if the gazebo and ros connections are ready
        """
        self._check_joint_states_ready()
        return True
    
    
    def _check_joint_states_ready(self):
        """
        Function to check if the joint states are received
        """
        ros_gazebo.gazebo_unpause_physics()
        # print( rostopic.get_topic_type(self.joint_state_topic, blocking=True))
        rospy.logdebug("Current "+ self.joint_state_topic +" READY")
            
        return True

    def _set_episode_init_params(self):
        """
        Sets the Robot in its init pose
        The Simulation will be unpaused for this purpose.
        """
        ros_gazebo.gazebo_set_model_state(self.model_name_in_gazebo, pos_x = self.model_pos_x, pos_y = self.model_pos_y, pos_z = self.model_pos_z)

        self.init_pos = np.array([0.0, -0.78, 1.5, 0.0, 0.8, 0.0])
        result = self.goal_joint_space_path_srv(self.init_pos)
        if not result:
            rospy.logwarn("Homing is failed....")

        #--- If training set random goal
        if self.training:
            # self.init_pos = self.get_randomJointVals()
            init_goal_vector = self.get_randomValidGoal()
            rospy.logwarn("Init goal: " + str(init_goal_vector))
            self.goal = init_goal_vector
            # init_goal_msg = SetLinkStateRequest()
            # init_goal_msg.link_state.pose.position.x = init_goal_vector[0]
            # init_goal_msg.link_state.pose.position.y = init_goal_vector[1]
            # init_goal_msg.link_state.pose.position.z = init_goal_vector[2]

            # self.set_init_goal_client.call(init_goal_msg)
            # self.goal = init_goal_vector
            rospy.logwarn("Desired goal--->" + str(self.goal))

        #--- Make Marker msg for publishing
        self.goal_marker.pose.position.x = self.goal[0]
        self.goal_marker.pose.position.y = self.goal[1]
        self.goal_marker.pose.position.z = self.goal[2]

        self.goal_marker.color.r = 1.0
        self.goal_marker.color.g = 0.0
        self.goal_marker.color.b = 0.0
        self.goal_marker.color.a = 1.0

        self.goal_marker.lifetime = rospy.Duration(secs=30)
        
        self.pub_marker.publish(self.goal_marker)

        rospy.logwarn("Initializing with values" + str(self.init_pos))
        result = self.goal_joint_space_path_srv(self.init_pos) 
        self.joint_angles = self.init_pos
        if not result:
            rospy.logwarn("Initialization is failed....")

    def _send_action(self, action): 
        """
        The action are the joint positions of manipulator and velocities of robomaster
        TODO Check what to do if movement result is False
        """

        # action = self.denormalize_action(action)
        # rospy.logwarn("=== Action: {}".format(action))
        self.action_state_prev = self.action_state
        self.action_state = action

        om_action = action[:self.n_actions_om]
        self.robomaster_action = action[self.n_actions_om:]

        # multiply 0.8 to the action of the robomaster angular z
        # robomaster_action[2] = robomaster_action[2]*0.8
        
        #--- Make actions as deltas
        om_action = om_action # + self.joint_values
        om_action = np.clip(om_action, self.min_joint_pos_values, self.max_joint_pos_values)

        # Log action components
        with open(self.action_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.action_fieldnames)
            writer.writerow({
                'timestep': self.timestep_counter,
                'joint1_action': om_action[0],
                'joint2_action': om_action[1],
                'joint3_action': om_action[2],
                'joint4_action': om_action[3],
                'joint5_action': om_action[4],
                'joint6_action': om_action[5],
                'robomaster_lin_x': self.robomaster_action[0],
                'robomaster_lin_y': self.robomaster_action[1],
                'robomaster_ang_z': self.robomaster_action[2]
            })

        # rospy.logwarn("Publishing om action: " + str(om_action))
        #--- Send actions to the robomaster (cmd_vel)
        self.robomaster_action = np.clip(self.robomaster_action, self.observations_robomaster_vel_min_range , self.observations_robomaster_vel_max_range)
        robomaster_twist = Twist()
        robomaster_twist.linear.x = self.robomaster_action[0]
        robomaster_twist.linear.y = self.robomaster_action[1]
        robomaster_twist.angular.z = self.robomaster_action[2]
        self.robomaster_action_pub.publish(robomaster_twist)

        # rospy.logwarn("Publishing robomaster action: " + str(robomaster_twist))
        self.movement_result = self.goal_joint_space_path_srv(om_action) 
        if not self.movement_result:
            # rospy.logerr("Movement result failed")
            self.stuck_open_manipulator = True
            return None
        else:
            self.stuck_open_manipulator = False
            # rospy.logwarn("Movement result success")
            current_time = rospy.get_time()
            time_diff = current_time - self.last_manipulator_action_time
            self.last_manipulator_action_time = current_time
            # rospy.logwarn(f"Time between previous action and this action: {time_diff} seconds")


    def denormalize_action(self, action):
        """Denormalize action from [-1, 1] back to real range."""
        return (action + 1) / 2 * (self._action_high - self._action_low) + self._action_low
    
    def normalize_observation(self, value, min_range, max_range):
        """ Normalize value to the range [-1, 1] """
        return 2 * (value - min_range) / (max_range - min_range) - 1

    
    def _get_observation(self):
        """
        Generate observations to be compatible with CustomCNNExtractor.
        Observations include:
        - "scan_matrix": A 2D laser scan image (64x64 matrix normalized between 0 and 1).
        - "vector": A 1D vector containing concatenated data like EE position, vector to goal, goal position, and joint values.
        """

        self.get_ee_pose()
        self.update_goal_and_publish_arrow()
        
        # --- Get current goal
        current_goal = self.goal

        # --- Get EE position
        # ee_pos_v = self.ee_pose  # Assuming this is a geometry_msgs/PoseStamped message
        
        # Handle case where ee_pose is None
        if self.ee_pose is None:
            rospy.logwarn("EE pose is None, using zero vector")
            self.ee_pose = np.zeros(3, dtype=np.float32)
        
        # --- Vector to goal
        vec_EE_GOAL = current_goal - self.ee_pose
        normalized_vec_EE_GOAL = vec_EE_GOAL / (np.linalg.norm(vec_EE_GOAL) + 1e-6)  # Add small epsilon to avoid division by zero
        # rospy.logwarn("Vector to goal: " + str(normalized_vec_EE_GOAL))

        # --- Scan matrix
        scan_matrix = self.scan_images_tensor  # Assuming this is already a 64x64 normalized tensor

        if scan_matrix is None:
            # rospy.logwarn("Scan matrix is None")
            scan_matrix = torch.zeros((64, 64))

        if self.joint_values is None:
            # rospy.logwarn("Joint values are None")
            self.joint_values = np.zeros(6, dtype=np.float32)

        if self.action_state_prev is None:
            # rospy.logwarn("Action state prev is None")
            self.action_state_prev = np.zeros(9, dtype=np.float32)
        
        action_state_prev_robomaster = self.action_state_prev[self.n_actions_om:]
        ee_to_goal_robomaster_action = self.ee_to_goal_robomaster_action
        
        

        # --- Create vector input (concatenated vector)
        vector = np.concatenate((
            normalized_vec_EE_GOAL,              # Vector from EE to Goal (normalized)
            current_goal,             # Position of Goal
            self.ee_pose,         # Current position of EE
            self.joint_values,         # Current joint angles
            action_state_prev_robomaster,  # Previous action state of Robomaster
            ee_to_goal_robomaster_action  # Standard action state of Robomaster
        ), axis=None).astype(np.float32)

        # Log observation vector components
        with open(self.obs_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.obs_fieldnames)
            writer.writerow({
                'timestep': self.timestep_counter,
                'vec_EE_GOAL_x': vec_EE_GOAL[0],
                'vec_EE_GOAL_y': vec_EE_GOAL[1],
                'vec_EE_GOAL_z': vec_EE_GOAL[2],
                'current_goal_x': current_goal[0],
                'current_goal_y': current_goal[1],
                'current_goal_z': current_goal[2],
                'ee_pose_x': self.ee_pose[0],
                'ee_pose_y': self.ee_pose[1],
                'ee_pose_z': self.ee_pose[2],
                'joint1': self.joint_values[0],
                'joint2': self.joint_values[1],
                'joint3': self.joint_values[2],
                'joint4': self.joint_values[3],
                'joint5': self.joint_values[4],
                'joint6': self.joint_values[5],
                'action_state_prev_robomaster_lin_x': action_state_prev_robomaster[0],
                'action_state_prev_robomaster_lin_y': action_state_prev_robomaster[1],
                'action_state_prev_robomaster_ang_z': action_state_prev_robomaster[2],
                'ee_to_goal_robomaster_lin_x': ee_to_goal_robomaster_action[0],
                'ee_to_goal_robomaster_lin_y': ee_to_goal_robomaster_action[1],
                'ee_to_goal_robomaster_ang_z': ee_to_goal_robomaster_action[2]
            })

        # Ensure vector input matches expected size
        assert vector.shape[0] == 21, f"Expected vector size of 21, but got {vector.shape[0]}"

        # rospy.logwarn("[robomaster_om_reacher] Shape of scan_matrix: " + str(scan_matrix.shape))
        self.time_count += 1
        # rospy.logwarn("[robomaster_om_reacher] Time count: " + str(self.time_count))
        # Construct the observation dictionary
        observation = {
            "scan_matrix": scan_matrix,
            "vector": vector
        }
        # rospy.logwarn("_get_observation is running!!!! ")
        return observation.copy()


    def _get_reward(self):
        """
        Given a success of the execution of the action
        Calculate the reward: binary => 1 for success, 0 for failure
        """
        current_pos = self.ee_pose
        reward = 0
        dist2goal = 0
        done = False
        prev_dist_reward = 0.0
        if self.goal is None or current_pos is None:
            done = False
            reward = 0
            return reward
        else: # TODO 이 부분은 어차피 _check_if_done에서 계산하니까 필요 없을 듯
            done = self.calculate_if_done(self.movement_result, self.goal, current_pos)

        if done:
            if self.pos_dynamic is False:
                # rospy.logwarn("SUCCESS Reached a Desired Position!")
                self.info['is_success'] = 1.0
            reward += self.reached_goal_reward
            self.goal_marker.header.stamp = rospy.Time.now()
            self.goal_marker.color.r = 0.0
            self.goal_marker.color.g = 1.0
            self.goal_marker.lifetime = rospy.Duration(secs=5)

        else:
            self.goal_marker.header.stamp = rospy.Time.now()
            self.goal_marker.color.r = 1.0
            self.goal_marker.color.g = 0.0
            self.goal_marker.lifetime = rospy.Duration(secs=5)
            self.goal_marker.header.frame_id="odom"

            dist2goal = scipy.spatial.distance.euclidean(current_pos, self.goal)
            # rospy.logwarn("Distance to goal: " + str(dist2goal))
            reward += - self.mult_dist_reward*dist2goal 
            reward += self.step_reward

            if self.prev_arrow_scale > self.arrow_scale:
                reward += 2*(self.prev_arrow_scale - self.arrow_scale) # 2/26 .5 -> 1.0 -> 1.5 3/6 2.5
                prev_dist_reward = 2*(self.prev_arrow_scale - self.arrow_scale)
                # rospy.logwarn("Arrow scale decreased -> Goal is closer")
            else:
                reward += 2*(self.prev_arrow_scale - self.arrow_scale)
                prev_dist_reward = 2*(self.prev_arrow_scale - self.arrow_scale)
                # rospy.logwarn("Arrow scale increased -> Goal is farther")

            # action_diff_weighted = 0.0  # Initialize the variable
            # if self.action_state_prev is not None and self.action_state is not None:
            #     action_diff = np.abs(self.action_state_prev - self.action_state)
            #     action_diff_weighted = -0.05*np.sum(action_diff) # 2/26 0.01 -> 0.1 -> 0.05
            #     reward += action_diff_weighted

        self.pub_marker.publish(self.goal_marker)

        # Penalize actions that are too different from the standard robomaster action
        # robomaster_action_diff = np.abs(self.ee_to_goal_robomaster_action - self.action_state[self.n_actions_om:])
        # reward += -1.5*np.sum(robomaster_action_diff) # 3/6 -1.5
            
        current_time = rospy.get_time()
        time_diff = current_time - self.last_reward_time
        self.last_reward_time = current_time
        # rospy.logwarn(f"Time between previous reward and this time step reward: {time_diff} seconds")

        # Log reward components
        distance_penalty = -(self.mult_dist_reward*dist2goal)
        goal_reward = done*self.reached_goal_reward

        obstacle_avoidance_reward = 0.5*np.exp(-0.9*self.min_distance)  
        reward += obstacle_avoidance_reward

        # rospy.logwarn(">>>REWARD>>>"+str(reward))
        # rospy.logwarn(">>>DISTANCE>>>"+str(distance_penalty))
        # rospy.logwarn(">>>IN_LIMITS>>>"+str(joint_limits_penalty))
        # rospy.logwarn(">>>STUCK_OPEN_MANIPULATOR>>>"+str(stuck_penalty))
        # rospy.logwarn(">>>DONE>>>"+str(goal_reward))
        # rospy.logwarn(">>>ACTION_DIFF>>>"+str(action_diff_weighted))
        # rospy.logwarn(">>>ROBOT_ACTION_DIFF>>>"+str(-np.sum(robomaster_action_diff)))

        # Write reward components to CSV
        self.timestep_counter += 1
        with open(self.reward_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.reward_fieldnames)
            writer.writerow({
                'timestep': self.timestep_counter,
                'total_reward': reward,
                'distance_penalty': distance_penalty,
                'prev_dist_reward': prev_dist_reward,
                'obstacle_avoidance_reward': obstacle_avoidance_reward,
                # 'joint_limits_penalty': joint_limits_penalty,
                # 'stuck_penalty': stuck_penalty,
                'goal_reward': goal_reward,
            })
        # rospy.logwarn("_get_reward is running!!!! ")
        return reward
    
    def _check_if_done(self):
        """
        Check if the EE is close enough to the goal or if the episode is done
        """
        if self.is_flipped or self.stuck_open_manipulator or self.is_stuck:
            rospy.logerr("-------Robot is stucked or robot is flipped. So Episode is done-------")
            done = True
            self.reward -= 300 # 2/27 -10000 -> -100 3/5 -100 -> -500 -> -300
            return done

        #--- Get current EE based on the observation
        #current_pos = observations['observation'][:3] # If using DICT
        current_pos = self.ee_pose # If using ARRAY
        # current_pos = np.array([current_pos.position.x, current_pos.position.y, current_pos.position.z])
        if self.goal is None or current_pos is None:
            # log the type of the goal and current_pos by rospy.logwarn
            rospy.logwarn("Type of goal: " + str(type(self.goal)))
            rospy.logwarn("Type of current_pos: " + str(type(current_pos)))
            rospy.logerr("Goal or EE position is None")
            done = False
            return done
        
        #--- Function used to calculate 
        done = self.calculate_if_done(self.movement_result, self.goal, current_pos)
        if done:
            rospy.logdebug("Reached a Desired Position!")

        #--- If the position is dynamic the episode is never done
        if self.pos_dynamic is True:
            done = False

        return done

    #-------------------------------------------------------#
    #  Internal methods for the UR5ReacherEnv         #

    def get_params(self):
        """
        get configuration parameters
        """
        
        self.sim_time = rospy.get_time()
        self.n_actions_om = rospy.get_param('/robomaster_om/n_actions_om')
        self.n_actions_robomaster = rospy.get_param('/robomaster_om/n_actions_robomaster')
        self.n_observations = rospy.get_param('/robomaster_om/n_observations')

        #--- Get parameter associated with ACTION SPACE

        self.min_joint_pos_values = rospy.get_param('/robomaster_om/min_joint_pos')
        self.max_joint_pos_values = rospy.get_param('/robomaster_om/max_joint_pos')

        self.min_joint_vel_values = rospy.get_param('/robomaster_om/min_joint_vel')
        self.max_joint_vel_values = rospy.get_param('/robomaster_om/max_joint_vel')

        self.robomaster_min_vel_value = rospy.get_param('/robomaster_om/robomaster_min_vel_value')
        self.robomaster_max_vel_value = rospy.get_param('/robomaster_om/robomaster_max_vel_value')

        assert len(self.min_joint_pos_values) == self.n_actions_om , "The min joint values do not have the same size as n_actions_om"
        assert len(self.max_joint_pos_values) == self.n_actions_om , "The max joint values do not have the same size as n_actions_om"

        assert len(self.min_joint_vel_values) == self.n_actions_om , "The min vel values do not have the same size as n_actions_om"
        assert len(self.max_joint_vel_values) == self.n_actions_om , "The max vel values do not have the same size as n_actions_om"

        assert len(self.robomaster_min_vel_value) == self.n_actions_robomaster , "The robomaster min vel values do not have the same size as n_actions_om"
        assert len(self.robomaster_max_vel_value) == self.n_actions_robomaster , "The robomaster max vel values do not have the same size as n_actions_om"

        #--- Get parameter associated with OBSERVATION SPACE

        self.position_ee_max = rospy.get_param('/robomaster_om/position_ee_max')
        self.position_ee_min = rospy.get_param('/robomaster_om/position_ee_min')
        self.position_goal_max = rospy.get_param('/robomaster_om/position_goal_max')
        self.position_goal_min = rospy.get_param('/robomaster_om/position_goal_min')
        self.max_distance = rospy.get_param('/robomaster_om/max_distance')

        #--- Get parameter asociated to goal tolerance
        self.tol_goal_ee = rospy.get_param('/robomaster_om/tolerance_goal_pos')
        self.training = rospy.get_param('/robomaster_om/training')
        self.pos_dynamic = rospy.get_param('/robomaster_om/pos_dynamic')
        # rospy.logwarn("Dynamic position:  " + str(self.pos_dynamic))

        #--- Get reward parameters
        self.reached_goal_reward = rospy.get_param('/robomaster_om/reached_goal_reward')
        self.step_reward = rospy.get_param('/robomaster_om/step_reward')
        self.mult_dist_reward = rospy.get_param('/robomaster_om/multiplier_dist_reward')
        self.joint_limits_reward = rospy.get_param('/robomaster_om/joint_limits_reward')
        self.arrow_scale_reward = rospy.get_param('/robomaster_om/arrow_scale_reward')

        #--- Get Gazebo physics parameters
        if rospy.has_param('/robomaster_om/time_step'):
            self.t_step = rospy.get_param('/robomaster_om/time_step')
            rospy.logwarn("Time step: " + str(self.t_step))
            ros_gazebo.gazebo_set_time_step(self.t_step)

        if rospy.has_param('/robomaster_om/update_rate_multiplier'):
            self.max_update_rate = rospy.get_param('/robomaster_om/update_rate_multiplier')
            rospy.logwarn("Max update rate: " + str(self.max_update_rate))
            ros_gazebo.gazebo_set_max_update_rate(self.max_update_rate)
        rospy.logwarn("get_params done")

    def get_elapsed_time(self):
        """
        Returns the elapsed time since the last check
        Useful to calculate rewards based on time
        """
        current_time = rospy.get_time()
        dt = self.sim_time - current_time
        self.sim_time = current_time
        return dt

    def test_goalPose(self, goal):
        """
        Function used to check if the defined goal is reachable
        """
        # rospy.logwarn("Goal to check: " + str(goal))
        result = self.check_goal(goal)
        '''
        def check_goal(self, goal):
            """
            Check if the goal is reachable
            * goal is a list with 3 elements, XYZ positions of the EE
            """
            result = self.move_robomaster_om_object.is_goal_reachable(goal) 
            return result
        --------------------------------------------------------------------------------
        def is_goal_reachable(self, goal):
            """
            Check if the goal is reachable
            * goal is the desired XYZ of the EE 
            """

            if isinstance(goal, type(np.array([0]))):
                goal = goal.tolist()

            goal[0] = float(goal[0])
            goal[1] = float(goal[1])
            goal[2] = float(goal[2])

            self.group.set_position_target(goal)
            plan = self.group.plan()        
            result = plan[0]
            self.group.clear_pose_targets()

            return result

        '''
        if result == False:
            rospy.logwarn( "The goal is not reachable")
        
        return result

    def get_randomValidGoal(self):
        
        goal = self.goal_space.sample()
        
        
        return goal

    def calculate_if_done(self, movement_result, goal, current_pos):
        """
        It calculated whether it has finished or not
        """
        done = False

        # If the previous movement was succesful
        if movement_result:
            rospy.logdebug("Movement was succesful")
        
        else:
            rospy.logwarn("Movement not succesful")

        # check if the end-effector located within a threshold to the goal
        distance_2_goal = scipy.spatial.distance.euclidean(current_pos, goal)

        if distance_2_goal<=self.tol_goal_ee:
            done = True
        
        return done

    # def goal_callback(self, data):
    #     """
    #     Callback to the topic used to send goals
    #     """
    #     self.goal = np.array([data.x, data.y, data.z])

    #     #--- Publish goal marker
    #     self.goal_marker.pose.position.x = self.goal[0]
    #     self.goal_marker.pose.position.y = self.goal[1]
    #     self.goal_marker.pose.position.z = self.goal[2]
    #     self.goal_marker.lifetime = rospy.Duration(secs=1)
    #     self.pub_marker.publish(self.goal_marker)

    def scan_image_callback(self, data): 
        """
        Callback to the topic scan_images and converts the image to a tensor
        """
        # rospy.logwarn("Scan image received")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono8") 
        except CvBridgeError as e:
            print(e)
            return
        
        np_image = np.array(cv_image)
        self.scan_images_tensor = torch.from_numpy(np_image).float() / 255.0
        # rospy.loginfo("Shape of the tensor: " + str(self.scan_images_tensor.shape))

    def scan_dist_check_callback(self, data):
        """
        Callback to the topic min_distance_threshold_breached
        """
        try:
            self.scan_dist_check_breached = data.data
        except:
            rospy.logwarn("Error in scan_dist_check_callback")
            rospy.logwarn(data)
            rospy.logwarn(data.data)

    def get_ee_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('odom', 'end_effector_link', rospy.Time(0), rospy.Duration(1.0)) # TODO Duration time
             # x, y, z and trans is TransformStamped object
            self.ee_pose = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr("TF Transform lookup failed")
            return None
        current_time = rospy.get_time()
        time_diff = current_time - self.last_get_ee_pose_time
        self.last_get_ee_pose_time = current_time
        # rospy.logwarn(f"Period of get_ee_pose: {time_diff} seconds\n")

    def get_goal_in_ee_frame(self):
        try:
            rospy.loginfo("Getting goal in EE frame")

            # self.listener.waitForTransform('/end_effector_link', '/odom', rospy.Time(0), rospy.Duration(0.005))
            # 'end_effector_link' 프레임을 기준으로 'odom'의 위치와 방향을 가져옴
            goal_odom = PointStamped()
            goal_odom.header.frame_id = '/odom'
            goal_odom.header.stamp = rospy.Time.now()
            # rospy.logwarn(f"Goal in odom frame: x: {self.goal[0]}, y: {self.goal[1]}, z: {self.goal[2]}")
            goal_odom.point.x = self.goal[0]
            goal_odom.point.y = self.goal[1]
            goal_odom.point.z = self.goal[2]

            goal_ee = self.listener.transformPoint('/end_effector_link', goal_odom)
            # rospy.loginfo(f"Goal in EE frame: x: {goal_ee.point.x}, y: {goal_ee.point.y}, z: {goal_ee.point.z}")
            self.goal_in_ee = np.array([goal_ee.point.x, goal_ee.point.y, goal_ee.point.z])
        
        except :
            rospy.logwarn("Error in get_goal_in_ee_frame")
            return None

        
    def goal_joint_space_path_srv(self, angles):
        """
        Service to send a trajectory to the robot
        """
        try:
            # Wait for the service with a timeout of 5 seconds
            # rospy.wait_for_service('/goal_joint_space_path', timeout=3.0)
            
            # Create a service client
            goal_joint_space_path_service_client = rospy.ServiceProxy('/goal_joint_space_path', SetJointPosition, persistent=True)
            goal_joint_space_path_request_object = SetJointPositionRequest()

            goal_joint_space_path_request_object.planning_group = 'arm'
            goal_joint_space_path_request_object.joint_position.joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            goal_joint_space_path_request_object.joint_position.position = [angles[0], angles[1], angles[2], angles[3], angles[4], angles[5]]
            goal_joint_space_path_request_object.path_time = 0.001

            # rospy.logwarn(f"goal_joint_space_path_request_object: {goal_joint_space_path_request_object}")
            # rospy.loginfo("Calling service...")
            result = goal_joint_space_path_service_client.call(goal_joint_space_path_request_object)
            
            # rospy.logwarn("Service call resultin goal_joint_space_path_srv: " + str(result))
            return result

        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            return False

        except rospy.ROSException as e:
            rospy.logerr("Service wait timeout: %s" % e)
            return False

    
    def joint_state_callback(self, data):
        """
        Callback to the topic joint_states
        """
        try:
            # slicing data.position to get only the joint values (joint1 to joint6)
            self.joint_values = np.array(data.position[2:8])

        except:
            rospy.logwarn("Error in joint_state_callback")
            rospy.logwarn(data)



    def update_goal_and_publish_arrow(self):
        """
        1. Transform the goal position from 'odom' frame to 'end_effector_link' frame
        2. Publish an arrow marker from the end effector to the goal
        """
        try:
            # rospy.loginfo("Updating goal in EE frame and publishing arrow marker")

            # self.goal이 올바르게 초기화되었는지 확인
            if not hasattr(self, 'goal') or self.goal is None or len(self.goal) < 3:
                rospy.logwarn("Goal position is not initialized properly!")
                return

            # 현재 최신 변환 정보 가져오기
            # trans, rot = self.listener.lookupTransform('/end_effector_link', '/odom', rospy.Time(0))

            # 'odom' 기준 목표 좌표 생성
            goal_odom = PointStamped()
            goal_odom.header.frame_id = '/odom'
            goal_odom.header.stamp = rospy.Time.now()
            # goal_odom.header.stamp = rospy.Time(0)
            goal_odom.point.x = self.goal[0]
            goal_odom.point.y = self.goal[1]
            goal_odom.point.z = self.goal[2]

            # rospy.logwarn(f"Goal in odom frame: x: {goal_odom.point.x}, y: {goal_odom.point.y}, z: {goal_odom.point.z}")

            # TODO 목표 지점을 'end_effector_link' 기준으로 변환 ->  지금 이 부분에서 transform이 안됨
            transform = self.tf_buffer.lookup_transform('end_effector_link', 'odom', rospy.Time(0), rospy.Duration(1.0))

            # 변환 적용
            goal_ee = do_transform_point(goal_odom, transform)

            # rospy.loginfo(f"Goal in EE frame: x: {goal_ee.point.x}, y: {goal_ee.point.y}, z: {goal_ee.point.z}")

            # 변환된 목표 좌표 저장
            self.goal_in_ee = np.array([goal_ee.point.x, goal_ee.point.y, goal_ee.point.z])

            # ===== 화살표 마커 생성 및 발행 =====
            arrow_marker = self.arrow_marker
            arrow_marker.header.stamp = rospy.Time.now()

            start_point = Point(x=0.0, y=0.0, z=0.0)  # 엔드 이펙터 기준 원점
            end_point = Point(x=self.goal_in_ee[0], y=self.goal_in_ee[1], z=self.goal_in_ee[2])  # 목표점

            arrow_marker.points = [start_point, end_point]
            # rospy.logwarn(f"Arrow points: {arrow_marker.points}")

            self.prev_arrow_scale = self.arrow_scale

            # 화살표 크기 조정
            arrow_length = np.linalg.norm([end_point.x, end_point.y, end_point.z])
            arrow_marker.scale.x = arrow_length * 0.05
            arrow_marker.scale.y = arrow_length * 0.1
            arrow_marker.scale.z = arrow_length * 0.1

            self.arrow_scale = (arrow_marker.scale.x + arrow_marker.scale.y + arrow_marker.scale.z) / 3
            # rospy.logwarn(f"Arrow scale: {self.arrow_scale}")
            
            # 마커 발행
            self.pub_arrow.publish(arrow_marker)
            # rospy.loginfo("Arrow marker published")
            direction = np.array([end_point.x, end_point.y])
        
            # 벡터 크기
            norm = np.linalg.norm(direction)

            if norm == 0 or np.isnan(norm):
                rospy.logwarn("Direction vector has zero magnitude or Nan!")
                return

            # 단위 벡터
            unit_vector = direction / norm

            # 목표 방향의 yaw 값 계산
            yaw = np.arctan2(unit_vector[1], unit_vector[0])
            # rospy.logwarn(f"Yaw: {yaw}")
            self.ee_to_goal_robomaster_action[0] = np.clip(unit_vector[0] * self.speed_scale, self.robomaster_min_vel_value["lin_x"], self.robomaster_max_vel_value["lin_x"])
            self.ee_to_goal_robomaster_action[1] = np.clip(unit_vector[1] * self.speed_scale, self.robomaster_min_vel_value["lin_y"], self.robomaster_max_vel_value["lin_y"])
            self.ee_to_goal_robomaster_action[2] = np.clip(yaw * self.speed_scale, self.robomaster_min_vel_value["ang_z"], self.robomaster_max_vel_value["ang_z"])
            # rospy.logwarn(f"Before clip EE to goal robomaster action: {unit_vector[0] * self.speed_scale}, {unit_vector[1] * self.speed_scale}, {yaw * self.speed_scale}")
            # rospy.logwarn(f"EE to goal robomaster action: {self.ee_to_goal_robomaster_action}")
            current_time = rospy.get_time()
            time_diff = current_time - self.last_update_goal_time
            self.last_update_goal_time = current_time
            # rospy.logwarn(f"Period of update_goal_and_publish_arrow: {time_diff} seconds\n")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("TF Transform lookup failed in update_goal_and_publish_arrow: " + str(e))
        except Exception as e:
            rospy.logwarn(f"Error in update_goal_and_publish_arrow: {e}")



    def imu_callback(self, data):
        quaternion = (
            data.orientation.x,
            data.orientation.y,
            data.orientation.z,
            data.orientation.w
        )

        euler = tf.transformations.euler_from_quaternion(quaternion)  
        roll, pitch, _ = euler  # 단위: 라디안

        # rospy.logerr(f"abs(roll): {abs(roll)}, abs(pitch): {abs(pitch)}")

        if abs(roll) > 1.57 or abs(pitch) > 1.57:  # 약 90도 초과
            rospy.logerr("-----------------Robot is flipped-----------------")
            self.is_flipped = True
        else:
            self.is_flipped = False

        self.is_robomaster_flipped_pub.publish(self.is_flipped)

        return
    
    # def odom_callback(self, data):
    #     odom_vel = np.array([data.twist.twist.linear.x, data.twist.twist.linear.y, data.twist.twist.angular.z])

    #     # check the self.robomaster_action and odom_vel is NoneType
    #     if self.robomaster_action is None or odom_vel is None:
    #         rospy.logwarn("robomaster_action or odom_vel is NoneType")
    #         return

    #     cmd_moving = abs(self.robomaster_action[0]) > 0.01 or abs(self.robomaster_action[1]) > 0.01 or abs(self.robomaster_action[2]) > 0.01
    #     odom_moving = abs(odom_vel[0]) > 0.01 or abs(odom_vel[1]) > 0.01 or abs(odom_vel[2]) > 0.01
    #     # odom_moving_stuck = abs(odom_vel[0]) < 0.02 and abs(odom_vel[1]) < 0.02 and abs(odom_vel[2]) < 0.02

    #     if cmd_moving and not odom_moving:
    #         rospy.logerr("-----------------Robomaster is stuck-----------------")
    #         rospy.logerr(f"linear x: {self.robomaster_action[0]}, linear y: {self.robomaster_action[1]}, angular z: {self.robomaster_action[2]}")
    #         rospy.logerr(f"odom_linear x: {odom_vel[0]}, odom_linear y: {odom_vel[1]}, odom_angular z: {odom_vel[2]}")
    #         self.is_stuck = True
    #     else:
    #         self.is_stuck = False

    #     self.is_robomaster_stuck_pub.publish(self.is_stuck)

    #     return
    
    def lidar_callback(self, data):
        """
        Callback to the topic lidar_scan
        """
        self.min_distance = min(data.ranges)
        # rospy.logerr(f"Closest obstacle distance: {self.min_distance}")
        if self.min_distance < 0.45:  
            self.is_stuck = True
            rospy.logerr("-----------------Robomaster is stuck-----------------")
            rospy.logerr(f"Closest obstacle distance: {self.min_distance}")
        else:
            self.is_stuck = False