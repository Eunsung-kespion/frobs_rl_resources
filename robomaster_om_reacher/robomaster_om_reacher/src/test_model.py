#!/usr/bin/python3.8
from robomaster_om_reacher.task_env.robomaster_om_reacher import Robomaster_OM_ReacherEnv

import gymnasium as gym
import rospy
import rospkg
import sys
from frobs_rl.common import ros_gazebo
from frobs_rl.common import ros_node
from frobs_rl.common import ros_params
from frobs_rl.wrappers.NormalizeActionWrapper import NormalizeActionWrapper
from frobs_rl.wrappers.TimeLimitWrapper import TimeLimitWrapper
from frobs_rl.wrappers.NormalizeObservWrapper import NormalizeObservWrapper

# Models
from frobs_rl.models.ddpg import DDPG
from frobs_rl.models.td3 import TD3
from frobs_rl.models.sac import SAC
from frobs_rl.models.ppo import PPO
from frobs_rl.models.dqn import DQN

if __name__ == '__main__':

    # Kill all processes related to previous runs
    ros_node.ros_kill_all_processes()
    # Kill all parameters
    ros_params.ros_kill_all_param()
    
    # Launch Gazebo 
    ros_gazebo.launch_Gazebo(paused=True, gui=True, use_sim_time=True, custom_world_path="/root/catkin_ws/src/aws-robomaker-small-warehouse-world/worlds/no_roof_small_warehouse.world")

    # Start node
    rospy.logwarn("Start")
    rospy.init_node('train_robomaster_om_reacher')

    # Launch the task environment
    env = gym.make('Robomaster_OM_ReacherEnv-v0')

    #--- Normalize action space
    env = NormalizeActionWrapper(env)

    #--- Normalize observation space
    # env = NormalizeObservWrapper(env)

    #--- Set max steps
    env = TimeLimitWrapper(env, max_steps=100)
    env.reset()

    #--- Set the save and log path
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("robomaster_om_reacher")

    #--- PPO
    save_model_path = pkg_path + "/models/static_reacher/sac/"
    
    model = SAC.load_trained(save_model_path + "20250313_010238/sac_model_1100000_steps.zip")

    obs = env.reset()
    obs = obs[0]
    episodes = 10
    epi_count = 0
    while epi_count < episodes:
        action, _states = model.predict(obs, deterministic=False)
        obs, _, dones, _, _ = env.step(action)
        rospy.logwarn("Action: " + str(action))
        # rospy.logwarn("Obs: " + str(obs))
        rospy.logwarn("Dones: " + str(dones))
        if dones:
            epi_count += 1
            rospy.logerr("Episode: " + str(epi_count))
            obs = env.reset()
            obs = obs[0]

    env.close()
    sys.exit()