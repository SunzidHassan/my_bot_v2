#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import pandas as pd
import cv2 as cv
import sys
# from pyntcloud import PyntCloud

class mybotsubpubNode(Node):
    def __init__(self) -> None:
        super().__init__("my_bot_subpub")
        self.get_logger().info("My bot subpub node has been started")
        self.cmd_vel_pub_ = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pose_subscriber = self.create_subscription(TFMessage, "/tf", self.tf_callback, 10)
        self.image_subscribe = self.create_subscription(CompressedImage, "/camera/image_raw/compressed", self.image_callback, 10)
        self.pointCloud_subscirbe = self.create_subscription(PointCloud2, "/camera/points", self.pointCloud_callback, 10)
        self.laser_subscribe = self.create_subscription(LaserScan, "/scan", self.laser_callback, 10)
        self.timer1 = self.create_timer(0.1, self.send_velocity_command)
        # self.timer2 = self.create_timer(0.1, self.saveObs)

        # save data into lists
        self.time = 0
        self.timeList = []
        self.posXlist = []
        self.posYlist = []
        self.angZlist = []
        self.angWlist = []
        self.linVelList = []
        self.angVelList = []


    def tf_callback(self, msg: TFMessage):
        tfSec = int(msg.transforms[0].header.stamp.sec)
        tfNanosec = int(msg.transforms[0].header.stamp.nanosec)
        if msg.transforms[0].header.frame_id == "odom":
            self.posX = round(msg.transforms[0].transform.translation.x, 2)
            self.posY = round(msg.transforms[0].transform.translation.y, 2)
            self.angZ = round(msg.transforms[0].transform.rotation.z, 2)
            self.angW = round(msg.transforms[0].transform.rotation.w, 2)

            if tfNanosec == 6000000:
                # self.get_logger().info("tfnano: " + str(tfNanosec))
                self.time += 1
                self.timeList.append(self.time)
                self.posXlist.append(self.posX)
                self.posYlist.append(self.posY)
                self.angZlist.append(self.angZ)
                self.angWlist.append(self.angW)
                self.saveObs()


    def image_callback(self, msg: CompressedImage):
        picSec = int(msg.header.stamp.sec)
        picnanosec = int(msg.header.stamp.nanosec)
        # self.get_logger().info("pictime: " + str(picnanosec))
        if picnanosec < 100000000:
            self.get_logger().info("picnano: " + str(picnanosec))
            image = np.asarray(bytearray(msg.data), dtype="uint8")
            self.imDecode = cv.imdecode(image, cv.IMREAD_COLOR)
            cv.imwrite(f'/ROS2_my_bot/my_bot/src/my_bot_controller/resource/24_10_19_sensorDump/egoCam/{self.time}.png', self.imDecode)


    def pointCloud_callback(self, msg: PointCloud2):
        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        # self.get_logger().info("Point cloud shape: " + str(points.shape))

        # cloud = PyntCloud(points)
        # cloud.to_file(f"/ROS2_my_bot/my_bot/src/my_bot_controller/resource/24_10_19_sensorDump/points/{self.posX}_{self.posY}_points.ply")

    def laser_callback(self, msg:LaserScan):
        self.laser = msg


    def send_velocity_command(self):
        msg = Twist() # create a msg object from Twist() class
        msg.linear.x = 0.01
        msg.angular.z = 0.05
        self.linVel = msg.linear.x
        self.angVel = msg.angular.z
        self.cmd_vel_pub_.publish(msg)

    
    def saveObs(self):
        self.get_logger().info("timer: " + str(self.time))
        if self.time > 20:
            data = {'time': self.timeList, 'x': self.posXlist, 'y': self.posYlist, 'z': self.angZlist, 'w': self.angWlist}
            df = pd.DataFrame(data)
            df.to_csv(f'/ROS2_my_bot/my_bot/src/my_bot_controller/resource/24_10_19_sensorDump/trajectory.csv', index=False)
            sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = mybotsubpubNode()
    rclpy.spin(node)
    rclpy.shutdown()