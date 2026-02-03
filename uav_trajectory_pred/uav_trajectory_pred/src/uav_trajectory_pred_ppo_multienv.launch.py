#!/bin/python3

import rospy
import roslaunch
import argparse

def generate_launch_description(num_envs):
    global env_launches
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)

    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()

    env_launches = []

    rospy.loginfo(f"Launching {num_envs} environments...")
    for i in range(num_envs):
        launch_file = "/root/catkin_ws/src/frobs_rl_resources/uav_trajectory_pred/uav_trajectory_pred/launch/uav_env.launch"
        args = [f"namespace:=env_{i}"]

        launch_arg = [(launch_file, args)]
        env_launch = roslaunch.parent.ROSLaunchParent(uuid, launch_arg)
        env_launch.start()
        env_launches.append(env_launch)  # 리스트에 추가해줘야 종료할 때 관리가 돼!

    train_node = roslaunch.core.Node(
        package="uav_trajectory_pred",
        node_type="train_ppo.py",
        name="train_uav_trajectory_pred",
        output="screen"
    )
    launch.launch(train_node)

    rospy.loginfo("Training node launched.")

    return launch

def shutdown_hook():
    global env_launches
    rospy.loginfo("Shutting down all launched environments...")
    for env in env_launches:
        rospy.loginfo(f"Shutting down {env}")
        env.shutdown()
    rospy.loginfo("All environments stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch UAV environments dynamically")
    parser.add_argument("--num_envs", type=int, default=3, help="Number of environments to launch")

    args = parser.parse_args()
    num_envs = args.num_envs

    rospy.init_node("uav_trajectory_launcher", anonymous=True)
    rospy.loginfo("Node is ready.")

    launch = generate_launch_description(num_envs)
    # 바로 여기에서 등록하는 게 좋아!
    # rospy.on_shutdown(shutdown_hook)

    rospy.logwarn("Press Ctrl+C to stop the environments")

    # spin은 ROS가 종료 신호를 받을 때까지 대기하는 역할
    rospy.spin()
