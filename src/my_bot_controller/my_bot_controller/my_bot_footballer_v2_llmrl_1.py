#!/usr/bin/env python3

# ros imports
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
import base64
import requests

# rl
# imports for torchrl
from collections import defaultdict
import pandas as pd
import torch
from tensordict.nn import TensorDictModule
from torch import nn


from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

import os

path = "/src/my_bot_controller/my_bot_controller"
os.chdir(path)

cwd = os.getcwd()
print(cwd)

# from my_bot_v2.src.my_bot_controller.my_bot_controller.my_bot_rl_v2.my_bot_rl_v2.my_bot_footballer_rl_v1 import my_botEnv
# from my_bot_rl_v2.my_bot_footballer_rl_v1 import my_botEnv
from my_bot_controller.my_bot_controller.my_bot_rl_v2.torchrl_1 import my_botEnv
baseEnv = my_botEnv()

net = nn.Sequential(
    nn.LazyLinear(64),
    nn.Tanh(),
    nn.LazyLinear(64),
    nn.Tanh(),
    nn.LazyLinear(64),
    nn.Tanh(),
    nn.LazyLinear(1),
)
policy = TensorDictModule(
    net,
    in_keys=["observation"],
    out_keys=["action"],
)

optim = torch.optim.Adam(policy.parameters(), lr=2e-3)

batch_size = 1
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 1000)
logs = defaultdict(list)

# imports for vlm
from transformers import AutoProcessor, AutoModelForCausalLM  
# Initialize the model and processor
model_id = 'microsoft/Florence-2-base'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

from my_bot_controller.my_bot_controller.my_bot_rl_v2.vlm_florence2_1 import segment_image

###

class mybotFootballerV2Node(Node):
    def __init__(self) -> None:
        super().__init__("my_bot_footballer_v2_llmrl")
        self.get_logger().info("My bot footballer node has been started")
        self.cmd_vel_pub_ = self.create_publisher(Twist, "/cmd_vel", 10)
        # self.timer = self.create_timer(0.5, self.send_velocity_command)
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
            imageArray = np.asarray(bytearray(msg.data), dtype="uint8")
            self.image = cv.imdecode(imageArray, cv.IMREAD_COLOR)
            # cv.imwrite(f'/my_bot/src/my_robot_controller/my_robot_controller/temp/LLMRun/{picTime}.png', self.imgSave)
            vlmOut = segment_image(self.image, self.prompt)
            self.segmentedImage = vlmOut[0]
            self.rewardVal = vlmOut[1]


    def pointCloud_callback(self, msg: PointCloud2): pass
        # pointTime = int(msg.header.stamp.sec)
        # self.get_logger().info("Time: " + str(pointTime))
        # points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        # self.generate_grid_map(points)
        # for point in points:
        #     self.get_logger().info(f'Point: {point}')


    def laser_callback(self, msg:LaserScan): pass
        # self.laser = msg


    def send_velocity_command(self, linVel=0.0, angVel=0.0):
        msg = Twist() # create a msg object from Twist() class
        msg.linear.x = linVel
        msg.angular.z = angVel
        self.cmd_vel_pub_.publish(msg)

    def rl(self, env=baseEnv):
        init_td = env.reset(env.gen_params(batch_size=[batch_size]))
        rollout = env.rollout(1, policy, tensordict=init_td, auto_reset=False)
        traj_return = rollout["next", "reward"].mean()
        (-traj_return).backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        self.get_logger().info(f"reward: {traj_return: 4.4f}, last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4}")
        logs["return"].append(traj_return.item())
        logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
        scheduler.step()


def main(args=None):
    rclpy.init(args=args)
    node = mybotFootballerV2Node()
    rclpy.spin(node)
    rclpy.shutdown()