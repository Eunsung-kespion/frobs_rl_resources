#!/bin/python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import Float32MultiArray

def matrix_callback(msg):
    """
    Callback to visualize the matrix received from the /scan_matrix topic.
    """
    # Extract dimensions from the message layout
    height = msg.layout.dim[0].size
    width = msg.layout.dim[1].size

    # Convert flat data to 2D matrix
    matrix = np.array(msg.data, dtype=np.float32).reshape((height, width))

    # Display the matrix using Matplotlib
    plt.imshow(matrix, cmap='viridis', origin='lower', extent=(0, width, 0, height))
    plt.colorbar(label="Range (meters)")
    plt.title("Laser Scan Matrix")
    plt.xlabel("X-axis (pixels)")
    plt.ylabel("Y-axis (pixels)")
    plt.pause(0.001)  # Allow real-time updating

def main():
    rospy.init_node('scan_matrix_visualizer', anonymous=True)
    rospy.Subscriber('/scan_matrix', Float32MultiArray, matrix_callback)
    plt.figure()  # Create a new Matplotlib figure
    rospy.spin()

if __name__ == '__main__':
    main()
