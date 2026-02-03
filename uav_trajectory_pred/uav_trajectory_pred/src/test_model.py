#!/bin/python3

import gymnasium as gym
import rospy
import rospkg
import sys
import time

import stable_baselines3
from stable_baselines3 import SAC

from uav_trajectory_pred.task_env.uav_trajectory_pred_sac_her import UAV_Trajectory_pred_SAC_HEREnv

def test_model(model_path, env_id, namespace="/", total_episodes=10, render=False):
    rospy.init_node('test_uav_trajectory_predEnv', anonymous=True)

    # 환경 초기화
    rospy.loginfo(f"Initializing environment: {env_id} with namespace: {namespace}")
    env = gym.make(env_id)

    # 시드 초기화
    env.reset(seed=0)

    # 모델 로드
    rospy.loginfo(f"Loading model from: {model_path}")
    model = SAC.load(model_path, env=env)

    # 에피소드별 테스트 루프
    for ep in range(total_episodes):
        rospy.loginfo(f"=== Episode {ep+1}/{total_episodes} ===")
        
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        time_out = False

        while not done and not time_out:
            action, _states = model.predict(obs, deterministic=True)  # 테스트는 deterministic=True 권장
            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            step += 1

            time_out = truncated

            # if render:
            #     env.render()  # 만약 환경에서 렌더링 지원하면 활성화
            # if step % 10 == 0:
            #     rospy.loginfo(f"Step: {step}, Reward: {reward}, Done: {done}, time_out: {time_out}")

            # 필요하면 sleep으로 속도 조절 가능
            time.sleep(0.01)

        rospy.logwarn(f"Episode {ep+1} finished! Total reward: {total_reward:.2f}")

    env.close()
    rospy.loginfo("All episodes completed.")

if __name__ == "__main__":
    # 테스트할 학습된 모델 경로
    model_path = "/root/catkin_ws/src/frobs_rl_resources/uav_trajectory_pred/uav_trajectory_pred/models/static_reacher/sac/sac_model_20100000_steps.zip"

    # 환경 ID
    env_id = "UAV_Trajectory_pred_SAC_HEREnv-v0"

    # 네임스페이스 지정 (기본적으로 "/")
    namespace = "/"

    # 에피소드 수와 렌더링 여부
    total_episodes = 5
    render = False  # True로 하면 render() 호출

    test_model(model_path, env_id, namespace=namespace, total_episodes=total_episodes, render=render)
