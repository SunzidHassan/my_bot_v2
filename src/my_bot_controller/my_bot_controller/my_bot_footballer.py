#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

import numpy as np
import cv2 as cv
import sensor_msgs_py.point_cloud2 as pc2
import re
from openai import OpenAI
import base64
import requests

###
# with open('/my_bot/src/my_robot_controller/resource/config.txt', 'r') as file:
#     api_key = file.read().strip()

api_key = ''
### Robot Prompt

delimiter = "#####"

task = """Given the robot view, select the best action for a mobile robot to move to a football goal post. Stop if the robot reaches very close to the goal post.
"""

actionInstructions = """
Action Selection Instruction 1: Move forward. (Action = 1)
Action Selection Instruction 2: Move right. (Action = 2).
Action Selection Instruction 3: Move left. (Action = 3).
Action Selection Instruction 4: Stop. (Action = 4).
"""

outputInstructions = f"""
Your response should use the following format:
<reasoning>
<reasoning>
<repeat until you have a decision>
Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation> 
Make sure to include {delimiter} to separate every step."""

robotPrompt = f"""
{delimiter} Task:
{task}
{delimiter} Action Selection Instructions:
{actionInstructions}
{delimiter} Output Instructions:
{outputInstructions}
"""

###

class mybotFootballerNode(Node):
    def __init__(self) -> None:
        super().__init__("my_bot_footballer")
        self.get_logger().info("My bot footballer node has been started")
        self.cmd_vel_pub_ = self.create_publisher(Twist, "/cmd_vel", 10)
        self.timer = self.create_timer(0.5, self.send_velocity_command)
        self.pose_subscriber = self.create_subscription(TFMessage, "/tf", self.tf_callback, 10)
        self.image_subscribe = self.create_subscription(CompressedImage, "/camera/image_raw/compressed", self.image_callback, 10)
        self.pointCloud_subscirbe = self.create_subscription(PointCloud2, "/camera/points", self.pointCloud_callback, 10)
        self.laser_subscribe = self.create_subscription(LaserScan, "/scan", self.laser_callback, 10)

        # self.grid_size = (500, 500)
        # self.grid_map = np.zeros(self.grid_size, dtype=np.uint8)

        self.actionID = 4

    def tf_callback(self, msg: TFMessage):
        if msg.transforms[0].header.frame_id == "odom": pass
            # self.get_logger().info("x: " + str(msg.transforms[0].transform.translation.x)\
            #                         +"\n" + "y: " + str(msg.transforms[0].transform.translation.y)\
            #                         +"\n" + "z: " + str(msg.transforms[0].transform.rotation.z)\
            #                         +"\n" + "w: " + str(msg.transforms[0].transform.rotation.w))

    def image_callback(self, msg: CompressedImage):
        picTime = int(msg.header.stamp.sec)
        # self.get_logger().info("time: " + str(picTime))
        if picTime % 5 == 0:
            image = np.asarray(bytearray(msg.data), dtype="uint8")
            self.imgSave = cv.imdecode(image, cv.IMREAD_COLOR)
            # cv.imwrite(f'/my_bot/src/my_robot_controller/my_robot_controller/temp/LLMRun/{picTime}.png', self.imgSave)
            self.imgDecode = base64.b64encode(image).decode('utf-8')
            self.LLMResponse = self.GPT4o()
            self.actionID = self.extractdAction()
            self.get_logger().info("actionID: " + str(self.actionID))
        


    def pointCloud_callback(self, msg: PointCloud2): pass
        # pointTime = int(msg.header.stamp.sec)
        # self.get_logger().info("Time: " + str(pointTime))
        # points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        # self.generate_grid_map(points)
        # for point in points:
        #     self.get_logger().info(f'Point: {point}')


    def laser_callback(self, msg:LaserScan): pass
        # self.laser = msg

    def GPT4o(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": robotPrompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self.imgDecode}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        return requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()['choices'][0]['message']['content']
    
    def extractdAction(self):
        pattern = r"#####\s*(\d+)"
        # Search for the pattern in the response text
        match = re.search(pattern, self.LLMResponse)
        return int(match.group(1))
    

    def send_velocity_command(self):
        msg = Twist() # create a msg object from Twist() class
        if self.actionID == 1: # move front
            msg.linear.x = 0.5
            msg.angular.z = 0.0
        elif self.actionID == 2: # move right
            msg.linear.x = 0.5
            msg.angular.z = 0.05
        elif self.actionID == 3: # move left
            msg.linear.x = 0.5
            msg.angular.z = -0.05
        elif self.actionID == 4: # stop
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        self.cmd_vel_pub_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = mybotFootballerNode()
    rclpy.spin(node)
    rclpy.shutdown()