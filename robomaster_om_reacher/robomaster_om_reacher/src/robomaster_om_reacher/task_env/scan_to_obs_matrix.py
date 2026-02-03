#!/usr/bin/python3
import rospy
import numpy as np
import torch
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError
import cv2
from scipy.ndimage import zoom

class LaserScanToMatrixConverter:
    def __init__(self):
        rospy.init_node('laser_scan_to_matrix_publisher')
        self.bridge = CvBridge()
        self.matrix_publisher = rospy.Publisher('/scan_matrix', Float32MultiArray, queue_size=10)
        self.image_publisher = rospy.Publisher('/scan_image', Image, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.callback)
        # rospy.Subscriber('/scan_image', Image, self.scan_image_callback, queue_size=10)

    def laser_scan_to_matrix(self, msg, square_size=8.0, resolution=0.1, target_size=(64, 64)):
        """
        Converts a LaserScan message into a binary grid matrix indicating scan data presence.
        """

        half_size = square_size / 2.0  # Half side length
        num_cells = int(square_size / resolution)  # Number of cells per side

        # Initialize the grid with 0 (no data)
        grid = np.ones((num_cells, num_cells), dtype=np.int8)
        origin_idx = num_cells // 2  # Center of the grid

        # Convert LaserScan data to Cartesian coordinates
        # rospy.logwarn("length of msg.ranges: %s", len(msg.ranges))
        for i, range_val in enumerate(msg.ranges):
            # Skip invalid readings
            if range_val < msg.range_min or range_val > msg.range_max:
                continue

            # Calculate angle for this measurement
            angle = msg.angle_min + i * msg.angle_increment

            # Convert polar to Cartesian coordinates
            x = range_val * np.cos(angle)
            y = range_val * np.sin(angle)

            # Ignore points outside the square
            if abs(x) > half_size or abs(y) > half_size:
                # rospy.logwarn("Point outside the square: x=%s, y=%s", x, y)
                continue

            # Map coordinates to grid indices
            grid_x = int((x + half_size) / resolution)
            grid_y = int((y + half_size) / resolution)

            # rospy.loginfo("grid_x: %s, grid_y: %s", grid_x, grid_y)

            # Mark the cell as "1" to indicate a scan hit
            grid[grid_y, grid_x] = 0

        # Resize the grid to the target size while preserving spatial structure
        zoom_factor = (target_size[0] / num_cells, target_size[1] / num_cells)
        resized_grid = zoom(grid, zoom_factor, order=0)  # Nearest-neighbor interpolation
        # rospy.loginfo("Resized grid shape: %s", resized_grid.shape)
        return resized_grid

    def matrix_to_image(self, matrix, height, width):
        """
        Converts a binary matrix to an OpenCV grayscale 2D image.
        """
        # Reshape the matrix to the target size
        matrix = np.array(matrix, dtype=np.uint8).reshape(height, width)

        # Scale the binary matrix (0 and 1) to 0-255 for visualization
        image = (matrix * 255).astype(np.uint8)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate for correct orientation
        image = cv2.flip(image, 1)  # Flip horizontally
        return image

    # def scan_image_callback(self, data):
    #     """
    #     Callback to the topic scan_images and converts the image to a tensor
    #     """
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
    #     except CvBridgeError as e:
    #         print(e)
    #         return

    #     np_image = np.array(cv_image)
    #     scan_images_tensor = torch.from_numpy(np_image).float() / 255.0
    #     # rospy.loginfo("Shape of the tensor: " + str(scan_images_tensor.shape))

    # for VIT extractor
    def callback(self, msg):
        """
        Callback function to process LaserScan messages, convert them to a binary matrix,
        resize to 224x224, and publish as Float32MultiArray and Image.
        """
        matrix = self.laser_scan_to_matrix(msg, target_size=(224, 224))

        message = Float32MultiArray()
        message.layout.dim.append(MultiArrayDimension(label="height", size=224, stride=224 * 224))
        message.layout.dim.append(MultiArrayDimension(label="width", size=224, stride=224))
        message.data = matrix.flatten().tolist()
        self.matrix_publisher.publish(message)

        image = self.matrix_to_image(matrix, height=224, width=224)
        ros_image = self.bridge.cv2_to_imgmsg(image, encoding="mono8")
        ros_image.header.stamp = msg.header.stamp
        self.image_publisher.publish(ros_image)


    # for CNN extractor
    # def callback(self, msg):
    #         """
    #         Callback function to process LaserScan messages, convert them to a binary matrix,
    #         resize to 64x64, and publish as Float32MultiArray and Image.
    #         """

    #         # Process the LaserScan message and convert it to a binary matrix
    #         matrix = self.laser_scan_to_matrix(msg, target_size=(64, 64))

    #         # Convert the binary matrix to Float32MultiArray for publishing
    #         message = Float32MultiArray()
    #         message.layout.dim.append(MultiArrayDimension(label="height", size=64, stride=64 * 64))
    #         message.layout.dim.append(MultiArrayDimension(label="width", size=64, stride=64))
    #         message.data = matrix.flatten().tolist()  # Flatten the matrix for publishing
    #         self.matrix_publisher.publish(message)

    #         # Convert the binary matrix to an OpenCV image
    #         image = self.matrix_to_image(matrix, height=64, width=64)

    #         # Convert the OpenCV image to a ROS Image message
    #         ros_image = self.bridge.cv2_to_imgmsg(image, encoding="mono8")
    #         ros_image.header.stamp = msg.header.stamp

    #         # Publish the Image message
    #         self.image_publisher.publish(ros_image)
    #         # rospy.loginfo("Published resized binary matrix as Image and Float32MultiArray")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    converter = LaserScanToMatrixConverter()
    converter.run()