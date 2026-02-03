#!/bin/python3

import gymnasium as gym
from gymnasium import utils
from gymnasium import spaces
from gymnasium.envs.registration import register
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
        id='UAV_Trajectory_pred_SAC_HEREnv-v0',
        entry_point='uav_trajectory_pred.task_env.uav_trajectory_pred_sac_her:UAV_Trajectory_pred_SAC_HEREnv',
        max_episode_steps=10000000,
        kwargs={'namespace': ''}
    )

class UAV_Trajectory_pred_SAC_HEREnv(gym.Env):
    """
    Custom Task Env, use this env to implement a task using the robot defined in the CustomRobotEnv
    """

    def __init__(self, namespace=''):
        """
        Describe the task.
        """
        rospy.logwarn("Starting UAV_Trajectory_pred_SAC_HEREnv Task Env")
        rospy.logwarn(f"Namespace: {namespace}")

        """
        Load YAML param file
        """
        # ros_params.ros_load_yaml_from_pkg("uav_trajectory_pred", "reacher_task.yaml", ns="/") 
        """
        Logging Setup
        """
        # Setup CSV logging for rewards
        self.namespace = namespace
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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

        self.timestep_counter = 0
        """
        Init necessary variables and objects.
        """
        self.goal_diff = []
    

        self.action_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.action_state_prev = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # for debugging
        self.time_count = 0
        self.dist_reward_scaling_factor = 0.5
        """
        Define the action and observation space.
        """

        #--- Action space
        self.goal_space = spaces.Box(
            low=np.array([8.5, -5.0, 0.0]), 
            high=np.array([9.0, 5.0, 5.0]), 
            dtype=np.float32)

        self.action_space = self.goal_space


        #--- Observation space -> Box type
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=np.array([8.5, -5.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([9.0, 5.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float32),
                shape=(6, ),
                dtype=np.float32
            ),
            "achieved_goal": spaces.Box(
                low=np.array([8.5, -5.0, 0.0]), 
                high=np.array([9.0, 5.0, 5.0]), 
                shape=(3, ),
                dtype=np.float32
            ),
            "desired_goal": spaces.Box(
                low=np.array([8.5, -5.0, 0.0]), 
                high=np.array([9.0, 5.0, 5.0]),  
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

        self.pub_marker = rospy.Publisher(f"{self.namespace}pred_result", Marker, queue_size=10)
        self.pub_start_env = rospy.Publisher(f"{self.namespace}start_env", Bool, queue_size=1)
        self.pub_resume_env = rospy.Publisher(f"{self.namespace}resume_env", Bool, queue_size=1)
        
        self.current_pose_sub = rospy.Subscriber(f"{self.namespace}current_point", Point, self.current_pose_callback)
        self.current_pose = None

        self.end_signal_sub = rospy.Subscriber(f"{self.namespace}end_signal", Bool, self.end_signal_callback)
        self.end_signal = False

        self.approx_goal_sub = rospy.Subscriber(f"{self.namespace}approx_goal", Point, self.approx_goal_callback)
        self.approx_goal = np.array([0.0, 0.0, 0.0])

        self.prev_pose = np.array([0.0, 0.0, 0.0])

        self.min_current_pos = np.array([0.0, -6.0, -6.0])
        self.max_current_pos = np.array([10.0, 6.0, 6.0])

        self.min_time_diff_pos = np.array([-1.0, -1.0, -1.0])
        self.max_time_diff_pos = np.array([1.0, 1.0, 1.0])

        self.end_episode = False

        rospy.logwarn(f"[{self.namespace}] Init super class")
        super(UAV_Trajectory_pred_SAC_HEREnv, self).__init__()
        
        
        """
        Finished __init__ method
        """
        rospy.logwarn(f"[{self.namespace}] Finished Init of UAV_Trajectory_pred_SAC_HEREnv Task Env")

    #-------------------------------------------------------#
    #   Custom available methods for the CustomTaskEnv      #


    def reset(self, seed=None):
        """
        Reset the environment
        """
        rospy.logwarn(f"[{self.namespace}] Resetting the UAV_Trajectory_pred_SAC_HEREnv Task Env")
        self._set_episode_init_params()
        return self._get_observation()

    def step(self, action):

        self._send_action(action)
        
        done = np.linalg.norm(self.action_state - self.approx_goal) < 0.05
        time_out = self._check_if_done()

        info ={
            "done": done,
            "TimeLimit.truncated": time_out
        }

        reward = self.compute_reward(self.action_state, self.approx_goal, info)

        # change reward [1,1] to scalar
        

        truncated = time_out

        self.end_episode = True
        self.end_signal = False

        obs, _ = self._get_observation()
        if time_out is True:
            rospy.logerr(f"In env, Episode ended due to time out")
        
        return obs, reward, done, truncated, info


    def _set_episode_init_params(self):
        """
        Initialize the Environment by publish the start_env signal
        """
        rospy.logerr(f"[{self.namespace}] Set the episode init params")
        
        self.accumulated_pos = np.array([])
        self.goal_diff = []
        
        self.pub_start_env.publish(bool(True))
        rospy.logerr(f"[{self.namespace}] Topic name: {self.namespace}start_env published")
        return True


    def _send_action(self, action): 
        """
        The action are the position of the UAV trajectory 
        TODO Check what to do if movement result is False
        """
        
        self.action_state_prev = self.action_state

        action = np.array(action, dtype=np.float32)
        self.action_state = action
        # rospy.loginfo("Action: {}".format(action))
        self.pred_result_marker.pose.position.x = action[0]
        self.pred_result_marker.pose.position.y = action[1]
        self.pred_result_marker.pose.position.z = action[2]

        self.pub_marker.publish(self.pred_result_marker)

        # with open(self.action_log_file, 'a', newline='') as f:
        #     writer = csv.DictWriter(f, fieldnames=self.action_fieldnames)
        #     writer.writerow({
        #         'timestep': self.timestep_counter,
        #         'action_x': action[0],
        #         'action_y': action[1],
        #         'action_z': action[2],
        #     })

    def _get_observation(self):
        """
        Get the observation from the environment
        Observations include:
        - "current_pos": current position of the UAV
        - "timediff_pos": time difference between the current position and the previous position
        """
        self.pub_resume_env.publish(bool(True))
        # rospy.loginfo(f"[{self.namespace}] Topic name: {self.namespace}resume_env published")
        current_pos = self.current_pose
        if current_pos is None:
            current_pos = np.array([0.0, 0.0, 0.0])
        
        if self.accumulated_pos.size == 0:
                self.accumulated_pos = current_pos.reshape(1, -1)
        else:
            self.accumulated_pos = np.concatenate((self.accumulated_pos, current_pos.reshape(1, -1)), axis=0)

        # rospy.logwarn("Shape of accumulated pos: {}".format(self.accumulated_pos.shape))

        time_diff_pos = self.prev_pose - current_pos
        
        observation = np.array([current_pos[0], current_pos[1], current_pos[2], time_diff_pos[0], time_diff_pos[1], time_diff_pos[2]], dtype=np.float32)

        # with open(self.obs_log_file, 'a', newline='') as f:
        #     writer = csv.DictWriter(f, fieldnames=self.obs_fieldnames)
        #     writer.writerow({
        #         'timestep': self.timestep_counter,
        #         'current_pos_x': current_pos[0],
        #         'current_pos_y': current_pos[1],
        #         'current_pos_z': current_pos[2],
        #         'timediff_pos_x': time_diff_pos[0],
        #         'timediff_pos_y': time_diff_pos[1],
        #         'timediff_pos_z': time_diff_pos[2],
        #     })
        

        info = {
            "done": False,
            "TimeLimit.truncated": False
        }

        obs = {
            "observation": observation,
            "achieved_goal": self.action_state,
            "desired_goal": self.approx_goal
        }
        
        return obs, info


    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Given a success of the execution of the action
        Calculate the reward: 1000.0 for success, 0 for failure, 1/(1+log(1+distance)) for intermediate rewards
        """

        
        # print("Achieved Goal: {}, Desired Goal: {}".format(achieved_goal, desired_goal))
        if achieved_goal is None:
            achieved_goal = np.array([0.0, 0.0, 0.0])
        self.goal_diff.append(np.linalg.norm(achieved_goal - desired_goal))

        achieved_goal = np.asarray(achieved_goal)
        desired_goal = np.asarray(desired_goal)

        # Handle step() case where achieved_goal is (3,) instead of (N,3)
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal.reshape(1, -1)  # Shape (1, 3)
        if desired_goal.ndim == 1:
            desired_goal = desired_goal.reshape(1, -1)  # Shape (1, 3)

        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1, keepdims=True)

        success = distance < 0.1
        # rospy.logerr("Done: {}, Success: {}".format(done, success))
        reward = 0.0
        self.timestep_counter += 1

        # Initialize reward array with zeros
        reward = np.zeros_like(distance)

        # Assign rewards based on success
        reward[success] = 10000.0  # Large reward for success

        # Failure case: Compute exponentially scaled failure reward
        if np.any(~success):  # If at least one goal was not successful
            # immediate_reward = 1 / (1 + np.log(1 + distance[~success]*100))  # for virtual sampling
            immediate_reward = np.exp(-0.1 * distance[~success])
            # rospy.logwarn("Immediate Reward: {}".format(immediate_reward))
            reward[~success] = immediate_reward
        if reward.shape == (1, 1):
            # rospy.logwarn(f"Shape of reward: {reward.shape}")
            reward = reward.squeeze()
            # rospy.logwarn(f"After squeeze, shape of reward: {reward.shape}")
            # [1,1] -> scalar

        return reward
    
    def _check_if_done(self):
        
        if self.end_signal:
            # rospy.logwarn(f"[{self.namespace}] In _check_if_done, End signal received")
            # self.end_episode = True
            # self.end_signal = False
            done = True
        else:
            done = False
        return done


    def current_pose_callback(self, msg):
        
        self.prev_pose = self.current_pose

        if self.prev_pose is None:
            self.prev_pose = np.array([0.0, 0.0, 0.0])

        self.current_pose = msg

        if self.current_pose is None:
            self.current_pose = np.array([0.0, 0.0, 0.0])
        else:
            self.current_pose = np.array([self.current_pose.x, self.current_pose.y, self.current_pose.z])
        
        # rospy.logwarn("Current Pose: {}".format(self.current_pose))

    def end_signal_callback(self, msg):
        self.end_signal = msg.data
        # rospy.logerr(f"[{self.namespace}] End signal received: {self.end_signal}")
        return
    
    def approx_goal_callback(self, msg):
        self.approx_goal = np.array([msg.x, msg.y, msg.z], dtype=np.float32)

        return