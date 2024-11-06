#!/usr/bin/env python3

# ros imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

from std_msgs.msg import Float64MultiArray
from gazebo_msgs.msg import ModelState, ModelStates, LinkStates
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty
import threading

import math
import numpy as np
import time
import copy
import random
import os
from datetime import datetime

import torch


#(x_agent, y_agent,
# x_ball, y_ball, z_ball,
# x_opponent, y_opponent)
observation = np.array([-0.75, 0,
                        0, 0, 1.0, 
                        0.75, 0], float)
previous_observation = np.array([0, 0,
                                 0, 0, 0, 
                                 0, 0], float)

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()

TIME_DELTA = 0.02
MAXLIVES = 5


class GazeboEnv(Node):
    def __init__(self):
        super().__init__('env')
        
        self.robot_vel1 = np.array([0,0], float)
        self.head_pos1 = np.array([0], float)
        
        self.set_state = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_world = self.create_client(Empty, "/reset_world")
        self.req = Empty.Request

        self.publisher_robot_vel1 = self.create_publisher(Float64MultiArray, '/robot_1/forward_velocity_controller/commands', 10)
        self.publisher_head_pos1 = self.create_publisher(Float64MultiArray, '/robot_1/forward_position_controller/commands', 10)
        
        #to relaunch sphere(ball)
        self.set_sphere_state = EntityState()
        self.set_sphere_state.name = "unit_sphere"
        self.set_sphere_state.pose.position.x = 0.0
        self.set_sphere_state.pose.position.y = 0.0
        self.set_sphere_state.pose.position.z = 0.175
        self.set_sphere_state.pose.orientation.x = 0.0
        self.set_sphere_state.pose.orientation.y = 0.0
        self.set_sphere_state.pose.orientation.z = 0.0
        self.set_sphere_state.pose.orientation.w = 1.0
        self.sphere_state = SetEntityState.Request()

        #to move a robot to initial position
        self.set_robot_1_state = EntityState()
        self.set_robot_1_state.name = "robot_1::robot_1_sliding_base_link"
        self.set_robot_1_state.pose.position.x = -0.75
        self.set_robot_1_state.pose.position.y = 0.0
        self.set_robot_1_state.pose.position.z = 0.115
        #self.set_robot_1_state.pose.orientation.x = 0.0
        #self.set_robot_1_state.pose.orientation.y = 0.0
        #self.set_robot_1_state.pose.orientation.z = 0.0
        #self.set_robot_1_state.pose.orientation.w = 1.0
        self.robot_1_state = SetEntityState.Request()             

        self.robot1_life = MAXLIVES

        self.t = 0
        self.t_limit = 6000

        #(x_agent, y_agent, x_agent_next, y_agent_next,
        # x_ball, y_ball, x_ball_next, y_ball_next,
        self.obs_robot1 = np.array([0, 0, -0.75, 0,
                                    0, 0, 0, 0, 
                                    0, 0, 0.75, 0], float)
        
        self.done = False

        self.robot_vel = 1.5
        self.move_head1 = False
        self.move_head1_t = 0.0

        self.t = 0
        self.t_limit = 6000

        #(x_agent, y_agent, x_agent_next, y_agent_next,
        # x_ball, y_ball, x_ball_next, y_ball_next,
        # x_opponent, y_opponent, x_opponent_next, y_opponent_next)
        self.obs_robot1 = np.array([0, 0, -0.75, 0,
                                    0, 0, 0, 0, 
                                    0, 0, 0.75, 0], float)
        

        self.done = False

        self.robot_vel = 1.5
        self.move_head1 = False
        self.move_head1_t = 0.0
