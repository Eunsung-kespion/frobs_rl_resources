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

import rospy


from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Twist, PointStamped
from std_msgs.msg import Bool


import numpy as np
import scipy.spatial

import csv
import os
from datetime import datetime

register(
        id='UAV_Trajectory_predEnv-v0',
        entry_point='uav_trajectory_pred.task_env.uav_trajectory_pred:UAV_Trajectory_predEnv',
        max_episode_steps=10000000,
        kwargs={'namespace': ''}
    )

class UAV_Trajectory_predEnv(robot_BasicEnv.RobotBasicEnv):
    """
    Custom Task Env, use this env to implement a task using the robot defined in the CustomRobotEnv
    """

    def __init__(self):
        """
        Describe the task.
        """
        rospy.logwarn("Starting UAV_Trajectory_predEnv Task Env")

        """
        Load YAML param file
        """
        # ros_params.ros_load_yaml_from_pkg("uav_trajectory_pred", "reacher_task.yaml", ns="/") 
        """
        Logging Setup
        """
        # Setup CSV logging for rewards
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # log_dir = os.path.join(os.path.dirname(__file__), '../logs/reward_logs')
        # os.makedirs(log_dir, exist_ok=True)
        # self.reward_log_file = os.path.join(log_dir, f'rewards_{timestamp}.csv')
        # self.reward_fieldnames = ['timestep', 'total_reward', 'goal_reached', 'failure_reward', 'immediate_reward']
        
        # with open(self.reward_log_file, 'w', newline='') as f:
        #     writer = csv.DictWriter(f, fieldnames=self.reward_fieldnames)
        #     writer.writeheader()

        # # Setup CSV logging for observations
        # obs_log_dir = os.path.join(os.path.dirname(__file__), '../logs/observation_logs')
        # os.makedirs(obs_log_dir, exist_ok=True)
        # self.obs_log_file = os.path.join(obs_log_dir, f'observations_{timestamp}.csv')
        # self.obs_fieldnames = [
        #     'timestep',
        #     'current_pos_x', 'current_pos_y', 'current_pos_z', 
        #     'timediff_pos_x', 'timediff_pos_y', 'timediff_pos_z'
        # ]
        
        # with open(self.obs_log_file, 'w', newline='') as f:
        #     writer = csv.DictWriter(f, fieldnames=self.obs_fieldnames)
        #     writer.writeheader()

        # # Setup CSV logging for actions
        # action_log_dir = os.path.join(os.path.dirname(__file__), '../logs/action_logs')
        # os.makedirs(action_log_dir, exist_ok=True)
        # self.action_log_file = os.path.join(action_log_dir, f'actions_{timestamp}.csv')
        # self.action_fieldnames = [
        #     'timestep',
        #     'action_x', 'action_y', 'action_z'
        # ]
        
        # with open(self.action_log_file, 'w', newline='') as f:
        #     writer = csv.DictWriter(f, fieldnames=self.action_fieldnames)
        #     writer.writeheader()

        # self.timestep_counter = 0
        """
        Init necessary variables and objects.
        """
    

        self.action_state = None
        self.action_state_prev = None
        self.min_distance = None

        # for debugging
        self.time_count = 0
        self.dist_reward_scaling_factor = 0.5
        """
        Define the action and observation space.
        """

        #--- Action space
        self.goal_space = spaces.Box(
            low=np.array([8.0, -5.0, 0.0]), 
            high=np.array([9.0, 5.0, 10.0]), 
            dtype=np.float32)

        self.action_space = self.goal_space


        #--- Observation space -> Box type
        # self.observation_space = spaces.Box(
        #     low=np.array([0.0, -6.0, -6.0, -1.0, -1.0, -1.0], dtype=np.float32),
        #     high=np.array([10.0, 6.0, 6.0, 1.0, 1.0, 1.0], dtype=np.float32),
        #     shape=(6, ),
        #     dtype=np.float32
        # )

        #--- Observation space -> Dict type (for MultiInputPolicy, HER)
        self.observation_space = spaces.Dict({
            'current_pos': spaces.Box(
                low=np.array([0.0, -6.0, -6.0], dtype=np.float32),
                high=np.array([10.0, 6.0, 6.0], dtype=np.float32),
                shape=(3, ),
                dtype=np.float32
            ),
            'timediff_pos': spaces.Box(
                low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                shape=(3, ),
                dtype=np.float32
            )
        })
       
        
        """
        Define subscribers or publishers as needed.
        """

        #--- Make Marker msg for publishing
        self.pred_result_marker = Marker()
        self.pred_result_marker.header.frame_id="world"
        self.pred_result_marker.header.stamp = rospy.Time.now()
        self.pred_result_marker.ns = "goal_shapes"
        self.pred_result_marker.id = 0
        self.pred_result_marker.type = Marker.SPHERE
        self.pred_result_marker.action = Marker.ADD

        self.pred_result_marker.pose.position.x = 0.0
        self.pred_result_marker.pose.position.y = 0.0
        self.pred_result_marker.pose.position.z = 0.0
        self.pred_result_marker.pose.orientation.x = 0.0
        self.pred_result_marker.pose.orientation.y = 0.0
        self.pred_result_marker.pose.orientation.z = 0.0
        self.pred_result_marker.pose.orientation.w = 1.0

        self.pred_result_marker.scale.x = 0.3
        self.pred_result_marker.scale.y = 0.3
        self.pred_result_marker.scale.z = 0.3

        # color like tennis ball 
        self.pred_result_marker.color.r = 0.8
        self.pred_result_marker.color.g = 1.0
        self.pred_result_marker.color.b = 0.2
        self.pred_result_marker.color.a = 1.0

        self.pub_marker = rospy.Publisher("pred_result", Marker, queue_size=10)
        self.pub_start_env = rospy.Publisher("/start_env", Bool, queue_size=1)
        self.pub_resume_env = rospy.Publisher("/resume_env", Bool, queue_size=1)
        
        self.current_pose_sub = rospy.Subscriber("current_point", Point, self.current_pose_callback)
        self.current_pose = None

        self.end_signal_sub = rospy.Subscriber("end_signal", Bool, self.end_signal_callback)
        self.end_signal = False

        self.approx_goal_sub = rospy.Subscriber("approx_goal", Point, self.approx_goal_callback)
        self.approx_goal = np.array([0.0, 0.0, 0.0])

        self.prev_pose = np.array([0.0, 0.0, 0.0])

        self.min_current_pos = np.array([0.0, -6.0, -6.0])
        self.max_current_pos = np.array([10.0, 6.0, 6.0])

        self.min_time_diff_pos = np.array([-1.0, -1.0, -1.0])
        self.max_time_diff_pos = np.array([1.0, 1.0, 1.0])

        self.end_episode = False
        
        reset_mode = 1
        step_mode = 1
        
        ros_gazebo.gazebo_set_time_step(0.01)

        self.use_sim_time = False
        self.model_name_in_gazebo = "robot"
        rospy.logwarn("Init super class")
        super(UAV_Trajectory_predEnv, self).__init__( launch_gazebo=False, spawn_robot=False, gazebo_freq = 1000, 
                    model_name_in_gazebo=self.model_name_in_gazebo, reset_mode=reset_mode, step_mode=step_mode, use_sim_time = self.use_sim_time)
        
        
        """
        Finished __init__ method
        """
        rospy.logwarn("Finished Init of UAV_Trajectory_predEnv Task Env")

    #-------------------------------------------------------#
    #   Custom available methods for the CustomTaskEnv      #

    def _check_subs_and_pubs_connection(self):
        """
        Function to check if the gazebo and ros connections are ready
        """
        return True


    def _set_episode_init_params(self):
        """
        Initialize the Environment by publish the start_env signal
        """
        rospy.logerr("Set the episode init params")
        ros_gazebo.gazebo_set_model_state(self.model_name_in_gazebo)
        
        self.accumulated_pos = np.array([])
        
        
        self.pub_start_env.publish(bool(True))
        # rospy.loginfo("Start the UAV trajectory node")

        return True


    def _send_action(self, action): 
        """
        The action are the position of the UAV trajectory 
        TODO Check what to do if movement result is False
        """
        
        self.action_state_prev = self.action_state
        self.action_state = action
        rospy.loginfo("Action: {}".format(action))
        self.pred_result_marker.pose.position.x = action[0]
        self.pred_result_marker.pose.position.y = action[1]
        self.pred_result_marker.pose.position.z = action[2]

        self.pub_marker.publish(self.pred_result_marker)

        with open(self.action_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.action_fieldnames)
            writer.writerow({
                'timestep': self.timestep_counter,
                'action_x': action[0],
                'action_y': action[1],
                'action_z': action[2],
            })

    def _get_observation(self):
        """
        Get the observation from the environment
        Observations include:
        - "current_pos": current position of the UAV
        - "timediff_pos": time difference between the current position and the previous position
        """
        self.pub_resume_env.publish(bool(True))
        current_pos = self.current_pose
        if current_pos is None:
            current_pos = np.array([0.0, 0.0, 0.0])
        
        if self.accumulated_pos.size == 0:
                self.accumulated_pos = current_pos.reshape(1, -1)
        else:
            self.accumulated_pos = np.concatenate((self.accumulated_pos, current_pos.reshape(1, -1)), axis=0)

        rospy.logwarn("Shape of accumulated pos: {}".format(self.accumulated_pos.shape))

        time_diff_pos = self.prev_pose - current_pos

        
        norm_current_pos = (current_pos - self.min_current_pos) / (self.max_current_pos - self.min_current_pos)

        norm_time_diff_pos = (time_diff_pos - self.min_time_diff_pos) / (self.max_time_diff_pos - self.min_time_diff_pos)

        observation = {
            'current_pos': norm_current_pos,
            'timediff_pos': norm_time_diff_pos
        }

        # observation = np.array([norm_current_pos[0], norm_current_pos[1], norm_current_pos[2], norm_time_diff_pos[0], norm_time_diff_pos[1], norm_time_diff_pos[2]])

        with open(self.obs_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.obs_fieldnames)
            writer.writerow({
                'timestep': self.timestep_counter,
                'current_pos_x': current_pos[0],
                'current_pos_y': current_pos[1],
                'current_pos_z': current_pos[2],
                'timediff_pos_x': time_diff_pos[0],
                'timediff_pos_y': time_diff_pos[1],
                'timediff_pos_z': time_diff_pos[2],
            })
        
        return observation.copy()


    def _get_reward(self):
        """
        Given a success of the execution of the action
        Calculate the reward: 1000.0 for success, 0 for failure, 1/(1+log(1+distance)) for intermediate rewards
        """
        success = self._check_if_success()
        # rospy.logerr("Done: {}, Success: {}".format(done, success))
        reward = 0.0
        self.timestep_counter += 1
        if success:
            self.info['is_success'] = 1.0
            rospy.logwarn("Success in episode")
            goal_reached = 10000.0
            reward += goal_reached
            
            with open(self.reward_log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.reward_fieldnames)
                writer.writerow({
                    'timestep': self.timestep_counter,
                    'total_reward': reward,
                    'goal_reached': goal_reached,
                    'failure_reward': 0.0,
                    'immediate_reward': 0.0,
                })

        elif self.end_episode and not success:
            rospy.logwarn("End Episode")
            self.end_episode = False
            failure_reward = 10000*np.exp(-self.min_distance)
            rospy.loginfo("Failure Reward: {}".format(failure_reward))
            reward += failure_reward

            with open(self.reward_log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.reward_fieldnames)
                writer.writerow({
                    'timestep': self.timestep_counter,
                    'total_reward': reward,
                    'goal_reached': 0.0,
                    'failure_reward': failure_reward,
                    'immediate_reward': 0.0,
                })

        elif self.action_state is not None and self.current_pose is not None:
            immediate_reward = 1 / (1 + np.log(1 + np.linalg.norm(self.action_state - self.approx_goal)))
            rospy.loginfo("Immediate Reward: {}".format(immediate_reward))
            reward += immediate_reward

            with open(self.reward_log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.reward_fieldnames)
                writer.writerow({
                    'timestep': self.timestep_counter,
                    'total_reward': reward,
                    'goal_reached': 0.0,
                    'failure_reward': 0.0,
                    'immediate_reward': immediate_reward,
                })

        else:
            with open(self.reward_log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.reward_fieldnames)
                writer.writerow({
                    'timestep': self.timestep_counter,
                    'total_reward': reward,
                    'goal_reached': 0.0,
                    'failure_reward': 0.0,
                    'immediate_reward': 0.0,
                })

        return reward
    
    def _check_if_done(self):
        
        if self.end_signal:
            rospy.logwarn("End signal received")
            self.end_episode = True
            self.end_signal = False
            done = True
        else:
            done = False
        return done

    def _check_if_success(self):
        rospy.loginfo(f"self.accumulated_pos.shape[0] : {self.accumulated_pos.shape[0]}")
        # rospy.loginfo(f"self.accumulated_pos: {self.accumulated_pos}")
        accumulated_pos_in_bound = (
                self.accumulated_pos[(self.accumulated_pos[:, 0] >= 8.0) & (self.accumulated_pos[:, 0] <= 9.0)]
                if self.accumulated_pos.shape[0] > 0
                else np.empty((0, 3))
            )
        # rospy.logerr("Accumulated Pos in Bound: {}".format(accumulated_pos_in_bound))
        rospy.loginfo("Shape of action state: {}".format(self.action_state.shape))
        distances = np.linalg.norm(accumulated_pos_in_bound - self.action_state, axis=1)

        # rospy.loginfo("Distances: {}".format(distances))
        rospy.loginfo("shape of distances: {}".format(distances.shape))
        
        if distances.size == 0:
            return False
        
        self.min_distance = np.min(distances)

        if self.min_distance <= 0.1:
                success = True
                
        else:
            success = False

        return success

    def current_pose_callback(self, msg):

        self.prev_pose = self.current_pose

        self.current_pose = msg

        if self.current_pose is None:
            self.current_pose = np.array([0.0, 0.0, 0.0])
        else:
            self.current_pose = np.array([self.current_pose.x, self.current_pose.y, self.current_pose.z])
        
        # rospy.logwarn("Current Pose: {}".format(self.current_pose))

    def end_signal_callback(self, msg):
        self.end_signal = msg.data

        return
    
    def approx_goal_callback(self, msg):
        self.approx_goal = np.array([msg.x, msg.y, msg.z])

        return