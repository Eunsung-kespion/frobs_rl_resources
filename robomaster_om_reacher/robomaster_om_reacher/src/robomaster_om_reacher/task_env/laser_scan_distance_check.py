#!/usr/bin/python3

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import math

class LaserScanDistanceCheck:
    def __init__(self):
        rospy.init_node('laser_scan_distance_check', anonymous=True)
        self.scan_subscriber = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.distance_publisher = rospy.Publisher("min_distance_threshold_breached", Bool, queue_size=10)
        self.distance_threshold = 0.3  # Set your desired distance threshold in meters
        rospy.loginfo("Distance threshold set to: %f meters", self.distance_threshold)

    def scan_callback(self, msg):
        min_distance = float('inf')  # Initialize with a very large value

        for i, range_val in enumerate(msg.ranges):
            if not math.isinf(range_val) and not math.isnan(range_val):
                angle = msg.angle_min + i * msg.angle_increment
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)
                distance = math.sqrt(x**2 + y**2)
                min_distance = min(min_distance, distance)

        threshold_breached = Bool()
        threshold_breached.data = min_distance < self.distance_threshold
        self.distance_publisher.publish(threshold_breached)
        # rospy.loginfo("Minimum distance: %f meters, Threshold breached: %s", min_distance, threshold_breached.data)

if __name__ == '__main__':
    try:
        laser_scan_distance_check = LaserScanDistanceCheck()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass