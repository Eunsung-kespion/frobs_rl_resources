#!/bin/python3

# from robomaster_om_reacher.task_env.robomaster_om_reacher import robomaster_om_moveit
from uav_trajectory_pred.task_env.uav_trajectory_pred_sac_her import UAV_Trajectory_pred_SAC_HEREnv
import gymnasium as gym
import rospy
import rospkg
import sys

from frobs_rl.common import ros_node
from frobs_rl.common import ros_params


import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback

import stable_baselines3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch as th

WANDB_PROJECT = rospy.get_param("/wandb_project_name", default="uav_trajectory_pred")
WANDB_ENTITY = rospy.get_param("/wandb_entity", default=None)

# Models
def get_policy_kwargs(ns="/"):
    """
    Function to get the policy kwargs from the ROS params server.

    @param ns: namespace of the ROS params server
    @type ns: str

    @return: policy kwargs
    """

    if rospy.get_param(ns + "/model_params/use_custom_policy") == True:
        # Activation function for the policy
        activation_function = rospy.get_param(ns + "/model_params/policy_params/activation_fn").lower()
        if activation_function == "relu":
            activation_fn = th.nn.ReLU
        elif activation_function == "tanh":
            activation_fn = th.nn.Tanh
        elif activation_function == "elu":
            activation_fn = th.nn.ELU
        elif activation_function == "selu":
            activation_fn = th.nn.SELU

        # Feature extractor for the policy
        feature_extractor = rospy.get_param(ns + "/model_params/policy_params/features_extractor_class")
        if feature_extractor == "FlattenExtractor":
            features_extractor_class = stable_baselines3.common.torch_layers.FlattenExtractor
        elif feature_extractor == "BaseFeaturesExtractor":
            features_extractor_class = stable_baselines3.common.torch_layers.BaseFeaturesExtractor
        elif feature_extractor == "CombinedExtractor":
            features_extractor_class = stable_baselines3.common.torch_layers.CombinedExtractor

        # Optimizer for the policy
        optimizer_class = rospy.get_param(ns + "/model_params/policy_params/optimizer_class")
        if optimizer_class == "Adam":
            optimizer_class = th.optim.Adam
        elif optimizer_class == "SGD":
            optimizer_class = th.optim.SGD
        elif optimizer_class == "RMSprop":
            optimizer_class = th.optim.RMSprop
        elif optimizer_class == "Adagrad":
            optimizer_class = th.optim.Adagrad
        elif optimizer_class == "Adadelta":
            optimizer_class = th.optim.Adadelta

        # Net Archiecture for the policy
        net_arch = rospy.get_param(ns + "/model_params/policy_params/net_arch")

        policy_kwargs = dict(activation_fn=activation_fn, features_extractor_class=features_extractor_class,
                            optimizer_class=optimizer_class, net_arch=net_arch)
        # rospy.logerr(f"policy_kwargs: {policy_kwargs}")
        # print(policy_kwargs)
    else:
        policy_kwargs = None

    return policy_kwargs

def make_env(env_id, procs ,seed=0):
    """각 환경별 네임스페이스를 설정하여 병렬 실행"""
    def _init():
        ns = f"/env_{procs}/"
        rospy.init_node(f"train_sac_her_env_{procs}", anonymous=True)  # 각 환경별 고유한 노드 실행
        env = gym.make(env_id, namespace=ns)  # 환경 실행 시 네임스페이스 전달
        print(f"Environment {env_id} initialized with namespace: {ns}")
        env.reset(seed =seed+procs)
        return env
    return _init


if __name__ == '__main__':

    # Kill all processes related to previous runs
    ros_node.ros_kill_all_processes()
    ros_params.ros_kill_all_param()
    
    # Launch Gazebo - use_sim_time=True
    # ros_gazebo.launch_Gazebo(paused=True, gui=True, use_sim_time=True, pub_clock_frequency = 100)
    # Launch Gazebo - use_sim_time=False
    # ros_gazebo.launch_Gazebo(paused=True, gui=True, use_sim_time=False, custom_world_path="/root/catkin_ws/src/aws-robomaker-small-warehouse-world/worlds/no_roof_small_warehouse.world")

    # Start node
    rospy.logwarn("Start")
    rospy.init_node('train_uav_trajectory_predEnv')

    num_envs = rospy.get_param("num_envs", 3)

    config_file_pkg = "uav_trajectory_pred"
    # config_filename = "sac.yaml"
    config_filename = "sac.yaml"
    ns = "/"
    ros_params.ros_load_yaml_from_pkg(config_file_pkg, config_filename, ns=ns)
    policy_kwargs = get_policy_kwargs(ns=ns)

    if rospy.get_param(ns + "/model_params/use_sde"):
        model_sde = True
        model_sde_sample_freq   = rospy.get_param(ns + "/model_params/sde_params/sde_sample_freq")
        model_use_sde_at_warmup = rospy.get_param(ns + "/model_params/sde_params/use_sde_at_warmup")
        action_noise = None
    else:
        model_sde = False
        model_sde_sample_freq   = -1
        model_use_sde_at_warmup = False

    model_learning_rate          = rospy.get_param(ns + "/model_params/sac_params/learning_rate")
    model_buffer_size            = rospy.get_param(ns + "/model_params/sac_params/buffer_size")
    model_learning_starts        = rospy.get_param(ns + "/model_params/sac_params/learning_starts")
    model_batch_size             = rospy.get_param(ns + "/model_params/sac_params/batch_size")
    model_tau                    = rospy.get_param(ns + "/model_params/sac_params/tau")
    model_gamma                  = rospy.get_param(ns + "/model_params/sac_params/gamma")
    model_gradient_steps         = rospy.get_param(ns + "/model_params/sac_params/gradient_steps")
    model_ent_coef               = rospy.get_param(ns + "/model_params/sac_params/ent_coef")
    model_target_update_interval = rospy.get_param(ns + "/model_params/sac_params/target_update_interval")
    model_target_entropy         = rospy.get_param(ns + "/model_params/sac_params/target_entropy")
    model_train_freq_freq        = rospy.get_param(ns + "/model_params/sac_params/train_freq/freq")
    model_train_freq_unit        = rospy.get_param(ns + "/model_params/sac_params/train_freq/unit")

    # Launch the task environment

    env_id = 'UAV_Trajectory_pred_SAC_HEREnv-v0'

    env = gym.make(env_id)
    # envs = SubprocVecEnv([make_env(env_id, i)for i in range(num_envs)], start_method="spawn")

    #--- Set the save and log path
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("uav_trajectory_pred")

    #-- SAC & HER
    policy="MultiInputPolicy"
    replay_buffer = "HerReplayBuffer"   
    save_path = pkg_path + "/models/static_reacher/sac/"
    log_path = pkg_path + "/logs/static_reacher/sac/"
    # model = SAC(env, save_path, log_path, config_file_pkg="uav_trajectory_pred", config_filename="sac.yaml", policy=policy, replay_buffer = replay_buffer)

    run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            sync_tensorboard=True,
            monitor_gym=True,  # automatically upload gym environments' videos
            config={
                # "algorithm": "SAC",
                'algorithm': 'SAC',
                "total_timesteps": rospy.get_param(ns + "/model_params/training_steps"),
                "learning_rate": rospy.get_param(ns + "/model_params/sac_params/learning_rate", default=0.0003),
                "n_steps": rospy.get_param(ns + "/model_params/sac_params/n_steps", default=100),
                "batch_size": rospy.get_param(ns + "/model_params/sac_params/batch_size", default=100),
                "n_epochs": rospy.get_param(ns + "/model_params/sac_params/n_epochs", default=5),
                "gamma": rospy.get_param(ns + "/model_params/sac_params/gamma", default=0.99),
            }
        )
    #--- Callback
    save_freq   = rospy.get_param(ns + "/model_params/save_freq")
    save_prefix = rospy.get_param(ns + "/model_params/save_prefix")
    checkpoint_callback = CheckpointCallback(  save_freq=save_freq, save_path=save_path,
                                                    name_prefix=save_prefix)
    
# Create a custom callback to monitor logged values
    class LogMonitorCallback(BaseCallback):
        def _on_step(self) -> bool:
            if self.n_calls % 100 == 0:  # Print every 100 steps
                metrics = self.logger.name_to_value
                # rospy.loginfo("Current metrics being logged:")
                for key, value in metrics.items():
                    rospy.loginfo(f"{key}: {value}")
            return True

    # Create WandbCallback with more metrics to track
    wandb_callback = WandbCallback(
    gradient_save_freq=100, model_save_path=f"models/{run.id}", verbose=2,
    log=[
        "train/episode_reward", "train/episode_length", "train/policy_loss", "train/value_loss",
        "train/entropy_loss", "train/  ", "rollout/success_rate",
        "train/learning_rate", "train/clip_fraction", "train/clip_range"
        ]
    )

    load_path = "/root/catkin_ws/src/frobs_rl_resources/uav_trajectory_pred/uav_trajectory_pred/models/static_reacher/sac/sac_model_5600000_steps.zip"


    callback_list = CallbackList([checkpoint_callback, wandb_callback, LogMonitorCallback()])

    replay_buffer_kwargs=dict(
                            n_sampled_goal=4,  # 샘플링할 목표 수
                            goal_selection_strategy="future",  # 목표 선택 전략 (future, final, episode 등)\
                            )
    model = stable_baselines3.SAC("MultiInputPolicy", env, verbose=1 ,action_noise=None,
                            use_sde=model_sde, sde_sample_freq=model_sde_sample_freq, use_sde_at_warmup=model_use_sde_at_warmup,
                            learning_rate=model_learning_rate, buffer_size=model_buffer_size, learning_starts=model_learning_starts,
                            batch_size=model_batch_size, tau=model_tau, gamma=model_gamma, gradient_steps=model_gradient_steps, 
                            policy_kwargs= policy_kwargs, ent_coef=model_ent_coef, target_update_interval=model_target_update_interval,
                            target_entropy=model_target_entropy, train_freq=(model_train_freq_freq, model_train_freq_unit), 
                            replay_buffer_class=stable_baselines3.HerReplayBuffer, replay_buffer_kwargs=replay_buffer_kwargs, tensorboard_log=log_path)

    # model = stable_baselines3.SAC.load(load_path, env=env, verbose=1, tensorboard_log=log_path, reset_num_timesteps=False)

    model.learn(total_timesteps=1000000000, log_interval=10000, tb_log_name="sac", reset_num_timesteps=False, callback=callback_list)

    wandb.finish()
    model.save(save_path + "final_model")