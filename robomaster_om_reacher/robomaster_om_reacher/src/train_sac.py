#!/bin/python3

# from robomaster_om_reacher.task_env.robomaster_om_reacher import robomaster_om_moveit
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

import wandb
from wandb.integration.sb3 import WandbCallback

# Models
from frobs_rl.models.ddpg import DDPG
from frobs_rl.models.td3 import TD3
from frobs_rl.models.sac import SAC
from frobs_rl.models.ppo import PPO
from frobs_rl.models.dqn import DQN

from datetime import datetime
import pytz

if __name__ == '__main__':

    # Kill all processes related to previous runs
    ros_node.ros_kill_all_processes()
    ros_params.ros_kill_all_param()
    
    # Launch Gazebo - use_sim_time=True
    ros_gazebo.launch_Gazebo(paused=True, gui=True, use_sim_time=True, custom_world_path="/root/catkin_ws/src/aws-robomaker-small-warehouse-world/worlds/no_roof_small_warehouse.world")
    # Launch Gazebo - use_sim_time=False
    # ros_gazebo.launch_Gazebo(paused=True, gui=True, use_sim_time=False, custom_world_path="/root/catkin_ws/src/aws-robomaker-small-warehouse-world/worlds/no_roof_small_warehouse.world")

    # Start node
    rospy.logwarn("Start")
    rospy.init_node('train_robomaster_om_reacher')

    # Launch the task environment
    env = gym.make('Robomaster_OM_ReacherEnv-v0')

    #--- Normalize action space
    env = NormalizeActionWrapper(env)

    #--- Normalize observation space
    env = NormalizeObservWrapper(env)

    #--- Set max steps
    # env = TimeLimitWrapper(env, max_steps=100)
    env = TimeLimitWrapper(env, max_steps=500)
    env.reset()

    #--- Set the save and log path
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("robomaster_om_reacher")

    # Choose the extractor policy
    # policy = "CustomCnnPolicy"
    policy = "CustomViTPolicy"

    #-- DDPG
    # save_path = pkg_path + "/models/static_reacher/ddpg/"
    # log_path = pkg_path + "/logs/static_reacher/ddpg/"
    # model = DDPG(env, save_path, log_path, config_file_pkg="abb_irb140_reacher", config_filename="ddpg.yaml")
    
    #-- TD3
    # save_path = pkg_path + "/models/td3/"
    # log_path = pkg_path + "/logs/td3/"
    # save_path = pkg_path + "/models/static_reacher/td3/"
    # log_path = pkg_path + "/logs/aux/td3/"
    # model = TD3(env, save_path, log_path, config_file_pkg="ur5_reacher", config_filename="td3.yaml")


    # Get the system time 
    korea_tz = pytz.timezone("Asia/Seoul")
    now = datetime.now(korea_tz)

    formatted_time = now.strftime("%Y%m%d_%H%M%S")
    
    #-- SAC
    save_path = pkg_path + "/models/static_reacher/sac/{}".format(formatted_time)
    log_path = pkg_path + "/logs/static_reacher/sac/{}".format(formatted_time)
    # save_path = pkg_path + "/models/static_reacher/sac/"
    # log_path = pkg_path + "/logs/static_reacher/sac/"
    model = SAC(env, save_path, log_path, config_file_pkg="robomaster_om_reacher", config_filename="sac.yaml", policy = policy)

    #--- PPO
    # save_path = pkg_path + "/models/static_reacher/ppo/"
    # log_path = pkg_path + "/logs/static_reacher/ppo/"
    # model = PPO(env, save_path, log_path, config_file_pkg="robomaster_om_reacher", config_filename="ppo.yaml", policy = policy)


    model.train()
    model.save_model()
    model.close_env()

    sys.exit()