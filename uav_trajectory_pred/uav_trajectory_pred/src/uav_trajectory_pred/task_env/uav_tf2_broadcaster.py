#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker
import math
import random
import numpy as np
from std_msgs.msg import Bool  # Bool 메시지 타입 추가

# 글로벌 변수 (True 신호 수신 여부)
global start_signal
global resume_signal
start_signal = False  
resume_signal = False

def bool_callback_start_env(msg):
    """ 특정 Bool 토픽의 메시지를 수신하여 start_signal을 변경 """
    global start_signal
    if msg.data:  # 메시지가 True일 때만 시작
        start_signal = True

def bool_callback_resume_env(msg):
    """ 특정 Bool 토픽의 메시지를 수신하여 resume_signal을 변경 """
    global resume_signal
    if msg.data:  # 메시지가 True일 때만 시작
        resume_signal = True

def broadcast_tf():
    global start_signal, resume_signal
    namespace = rospy.get_namespace().strip("/")  

    rospy.init_node(f'{namespace}_uav_tf2_broadcaster')
    rospy.loginfo(f" [{namespace}] Start UAV TF2 Broadcaster")
    
    

    br = tf2_ros.TransformBroadcaster()
    rate = rospy.Rate(100)  # 100Hz (0.01초마다 실행)

    current_point_pub = rospy.Publisher('current_point', Point, queue_size=1)
    end_signal_pub = rospy.Publisher('end_signal', Bool, queue_size=1)
    approx_goal_pub = rospy.Publisher('approx_goal', Point, queue_size=1)
    approx_goal_marker_pub = rospy.Publisher('approx_goal_marker', Marker, queue_size=1)


    rospy.Subscriber("start_env", Bool, bool_callback_start_env)
    rospy.Subscriber("resume_env", Bool, bool_callback_resume_env)

    approx_goal_marker = Marker()
    approx_goal_marker.header.frame_id = "world"
    approx_goal_marker.header.stamp = rospy.Time.now()
    approx_goal_marker.ns = "approx_goal"
    approx_goal_marker.id = 0
    approx_goal_marker.type = Marker.SPHERE
    approx_goal_marker.action = Marker.ADD

    rendezvous_time = 440.0

    while not rospy.is_shutdown():
        # 신호가 올 때까지 대기
        if not start_signal:
            rospy.loginfo_throttle(5, f"[{namespace}] Waiting for start_env signal...")
            rate.sleep()
            continue  # 다음 루프 실행
        
        # 시작 위치 & 도착 위치
        start_x, start_y, start_z = 0, 0, 1
        end_x, end_z = 10, 0
        end_y = random.uniform(-5.0, 5.0)  # -10.0 ~ 10.0 사이 랜덤 y 값

        approx_x = start_x + (end_x - start_x) * (rendezvous_time / 500.0) 
        approx_y = start_y + (end_y - start_y) * (0.5 - 0.5 * math.cos(math.pi * (rendezvous_time / 500.0))) 
        approx_z = (-1/50625) * (rendezvous_time - 50)**2 + 4

        approx_goal = np.array([approx_x, approx_y, approx_z])
        approx_goal_pub.publish(Point(approx_goal[0], approx_goal[1], approx_goal[2]))

        approx_goal_marker.pose.position.x = approx_x
        approx_goal_marker.pose.position.y = approx_y
        approx_goal_marker.pose.position.z = approx_z
        approx_goal_marker.scale.x = 0.3
        approx_goal_marker.scale.y = 0.3
        approx_goal_marker.scale.z = 0.3
        # color yellow
        approx_goal_marker.color.r = 1.0
        approx_goal_marker.color.g = 0.0
        approx_goal_marker.color.b = 0.0
        approx_goal_marker.color.a = 1.0
        approx_goal_marker_pub.publish(approx_goal_marker)

        t = 0
        while t <= rendezvous_time - 30 and not rospy.is_shutdown():
            if not resume_signal:
                rospy.loginfo_throttle(5, f"[{namespace}] Waiting for resume_env signal...")
                rate.sleep()

            current_point = Point() 
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = f"{namespace}/world"
            transform.child_frame_id = f"{namespace}/uav"
            # rospy.loginfo(f"transform.header.frame_id: {transform.header.frame_id}")
            # rospy.loginfo(f"transform.child_frame_id: {transform.child_frame_id}")
            # 정규분포 노이즈 추가 (N(0, 0.02))
            noise_x = random.gauss(0, 0.02)
            noise_y = random.gauss(0, 0.02)
            noise_z = random.gauss(0, 0.02)

            # x는 선형 증가
            x = start_x + (end_x - start_x) * (t / 500.0) + noise_x
            # y는 부드러운 곡선을 따라 이동 (cos 함수 이용)
            y = start_y + (end_y - start_y) * (0.5 - 0.5 * math.cos(math.pi * (t / 500.0))) + noise_y
            # z는 포물선 경로
            z = (-1/50625) * (t - 50)**2 + 4 + noise_z


            transform.transform.translation.x = x
            transform.transform.translation.y = y
            transform.transform.translation.z = z
            transform.transform.rotation.w = 1.0  # 기본 회전값 설정

            br.sendTransform(transform)

            current_point.x = x
            current_point.y = y
            current_point.z = z

            current_point_pub.publish(current_point)
            # rospy.logwarn(f"[{namespace}] Step: {t}, x: {x}, y: {y}, z: {z}")
            t += 1
            resume_signal = False
            rate.sleep()

        # 한 번 루프 실행 후 신호 초기화 (다음 신호 대기)
        start_signal = False
        end_signal_pub.publish(Bool(True))
        rospy.logerr(f"[{namespace}] Movement completed. Waiting for next start signal.")

if __name__ == '__main__':
    try:
        broadcast_tf()
    except rospy.ROSInterruptException:
        pass
