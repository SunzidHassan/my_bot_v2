#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import math
from std_msgs.msg import Float64MultiArray
from gazebo_msgs.msg import ModelState, ModelStates, LinkStates
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty
import threading
import numpy as np
import time
import copy
import random
import torch
import os
from datetime import datetime

from PPO import PPO

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
        self.robot_vel2 = np.array([0,0], float)
        self.head_pos1 = np.array([0], float)
        self.head_pos2 = np.array([0], float)
        
        self.set_state = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_world = self.create_client(Empty, "/reset_world")
        self.req = Empty.Request

        self.publisher_head_pos1 = self.create_publisher(Float64MultiArray, '/robot_1/forward_position_controller/commands', 10)
        self.publisher_head_pos2 = self.create_publisher(Float64MultiArray, '/robot_2/forward_position_controller/commands', 10)
        self.publisher_robot_vel1 = self.create_publisher(Float64MultiArray, '/robot_1/forward_velocity_controller/commands', 10)
        self.publisher_robot_vel2 = self.create_publisher(Float64MultiArray, '/robot_2/forward_velocity_controller/commands', 10)
        
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
        self.robot2_life = MAXLIVES

        self.t = 0
        self.t_limit = 6000

        #(x_agent, y_agent, x_agent_next, y_agent_next,
        # x_ball, y_ball, x_ball_next, y_ball_next,
        # x_opponent, y_opponent, x_opponent_next, y_opponent_next)
        self.obs_robot1 = np.array([0, 0, -0.75, 0,
                                    0, 0, 0, 0, 
                                    0, 0, 0.75, 0], float)
        
        self.obs_robot2 = np.array([0, 0, -0.75, 0,
                                    0, 0, 0, 0,
                                    0, 0, 0.75, 0], float)

        self.done = False

        self.robot_vel = 1.5
        self.move_head1 = False
        self.move_head2 = False
        self.move_head1_t = 0.0
        self.move_head2_t = 0.0

    def step(self, robot1_action, robot2_action=None):
        global observation, previous_observation

        self.t += 1

        ## agent1 (robot1)
        # 0: forward 
        # 1: backward 
        # 2: punch 
        # 3: left
        # 4: right
        # forth back
        if(robot1_action == 0): #forth
            if(observation[0] < -0.05):
                self.robot_vel1[0] = 1.0
            else:
                self.robot_vel1[0] = 0.0
        elif(robot1_action == 1): #back
            if(observation[0] > -1.47):
                self.robot_vel1[0] = -1.0
            else:
                self.robot_vel1[0] = 0.0

        elif(robot1_action == 2): #punch
            if(self.move_head1 == False):
                self.move_head1 = True

        elif(robot1_action == 3): #left
            if(observation[1] < 0.68):
                self.robot_vel1[1] = 1.0
            else:
                self.robot_vel1[1] = 0.0
        elif(robot1_action == 4): #right
            if(observation[1] > -0.68):
                self.robot_vel1[1] = -1.0
            else:
                self.robot_vel1[1] = 0.0

        if(self.move_head1 == True):
            self.head_pos1[0] = 0.05*math.sin(24*self.move_head1_t)
            if(self.move_head1_t >= math.pi/24):
                self.move_head1_t = 0  
                self.move_head1 = False
            self.move_head1_t += TIME_DELTA 
        else:
            self.head_pos1[0] = 0
            self.move_head1_t = 0

        # Publishing the robot action
        ## agent2 (robot2)
        if(robot2_action == 0): #forth
            if(observation[5] > 0.05):
                self.robot_vel2[0] = 1.0
            else:
                self.robot_vel2[0] = 0.0                
        elif(robot2_action == 1): #back
            if(observation[5] < 1.47):
                self.robot_vel2[0] = -1.0
            else:
                self.robot_vel2[0] = 0.0 
        elif(robot2_action == 2): #punch
            if(self.move_head2 == False):
                self.move_head2 = True
        elif(robot2_action == 3): #left
            if(observation[6] > -0.68):
                self.robot_vel2[1] = 1.0
            else:
                self.robot_vel2[0] = 0.0                              
        elif(robot2_action == 4): #right
            if(observation[6] < 0.68):
                self.robot_vel2[1] = -1.0
            else:
                self.robot_vel2[0] = 0.0 

        if(self.move_head2 == True):
            self.head_pos2[0] = 0.05*math.sin(24*self.move_head2_t)
            if(self.move_head2_t >= math.pi/24):
                self.move_head2_t = 0  
                self.move_head2 = False
            self.move_head2_t += TIME_DELTA 
        else:
            self.head_pos2[0] = 0
            self.move_head2_t = 0

        #publish robot1 commands
        array_forPublish1_head = Float64MultiArray(data=self.head_pos1) 
        self.publisher_head_pos1.publish(array_forPublish1_head)
        array_forPublish1_vel = Float64MultiArray(data=self.robot_vel1) 
        self.publisher_robot_vel1.publish(array_forPublish1_vel)

        #publish robot2 commands
        array_forPublish2_head = Float64MultiArray(data=self.head_pos2) 
        self.publisher_head_pos2.publish(array_forPublish2_head)
        array_forPublish2_vel = Float64MultiArray(data=self.robot_vel2) 
        self.publisher_robot_vel2.publish(array_forPublish2_vel) 

        while not gz_env.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.unpause.call_async(Empty.Request())
        except:
            self.get_logger().info("/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        try:
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e:
            self.get_logger().info("/gazebo/pause_physics service call failed")

        # the ball is out of the boundary
        if(abs(observation[2]) > 1.55):
            if(observation[2] < -1.55):
                self.robot1_life -= 1
                reward = -1
                self.get_logger().info('ROBOT1 LOST A POINT!')
            elif(observation[2] > 1.55):
                self.robot2_life -= 1
                reward = 1
                self.get_logger().info('ROBOT1 GET A POINT!')
            self.reset()
        else:
            reward = 0

        if self.t >= self.t_limit:
            #self.get_logger().info('TIME LIMIT!')
            self.done = True

        if self.robot1_life <= 0 or self.robot2_life <= 0:
            #self.get_logger().info(f'LIFE LIMIT! robot1:{self.robot1_life} robot2:{self.robot2_life}')
            self.done = True

        self.obs_robot1[0] = previous_observation[0]
        self.obs_robot1[1] = previous_observation[1]
        self.obs_robot1[2] = observation[0]
        self.obs_robot1[3] = observation[1]
        self.obs_robot1[4] = previous_observation[2]
        self.obs_robot1[5] = previous_observation[3]
        self.obs_robot1[6] = observation[2]
        self.obs_robot1[7] = observation[3]
        self.obs_robot1[8] = previous_observation[5]
        self.obs_robot1[9] = previous_observation[6]
        self.obs_robot1[10] = observation[5]
        self.obs_robot1[11] = observation[6]

        self.obs_robot2[0] = -self.obs_robot1[8]
        self.obs_robot2[1] = -self.obs_robot1[9]
        self.obs_robot2[2] = -self.obs_robot1[10]
        self.obs_robot2[3] = -self.obs_robot1[11]
        self.obs_robot2[4] = -self.obs_robot1[4]
        self.obs_robot2[5] = -self.obs_robot1[5]
        self.obs_robot2[6] = -self.obs_robot1[6]
        self.obs_robot2[7] = -self.obs_robot1[7]
        self.obs_robot2[8] = -self.obs_robot1[0]
        self.obs_robot2[9] = -self.obs_robot1[1]
        self.obs_robot2[10] = -self.obs_robot1[2]
        self.obs_robot2[11] = -self.obs_robot1[3]

        previous_observation = copy.copy(observation)

        return self.obs_robot1, self.obs_robot2, reward, self.done

    def reset(self):

        while not self.reset_world.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            #self.get_logger().info('Resetting the world')
            self.reset_world.call_async(Empty.Request())
        except:
            import traceback
            traceback.print_exc()

        if(self.done):
            self.robot1_life = MAXLIVES
            self.robot2_life = MAXLIVES
            self.t = 0
            self.done = False

        self.robot_1_state = SetEntityState.Request()
        self.robot_1_state._state = self.set_robot_1_state
        while not self.set_state.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            self.set_state.call_async(self.robot_1_state)
        except rclpy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        self.set_sphere_state.twist.linear.x = -1*(0.3 + 0.6*random.random()) 
        self.set_sphere_state.twist.linear.y = 0.6*(random.random()-0.5)
        self.set_sphere_state.twist.linear.z = 0.0

        # relaunch the ball
        self.sphere_state._state = self.set_sphere_state
        while not self.set_state.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')
        try:
            self.set_state.call_async(self.sphere_state)
        except:
            import traceback
            traceback.print_exc()

        self.obs_robot1[0] = previous_observation[0]
        self.obs_robot1[1] = previous_observation[1]
        self.obs_robot1[2] = observation[0]
        self.obs_robot1[3] = observation[1]
        self.obs_robot1[4] = previous_observation[2]
        self.obs_robot1[5] = previous_observation[3]
        self.obs_robot1[6] = observation[2]
        self.obs_robot1[7] = observation[3]
        self.obs_robot1[8] = previous_observation[5]
        self.obs_robot1[9] = previous_observation[6]
        self.obs_robot1[10] = observation[5]
        self.obs_robot1[11] = observation[6]

        self.obs_robot2[0] = -self.obs_robot1[8]
        self.obs_robot2[1] = -self.obs_robot1[9]
        self.obs_robot2[2] = -self.obs_robot1[10]
        self.obs_robot2[3] = -self.obs_robot1[11]
        self.obs_robot2[4] = -self.obs_robot1[4]
        self.obs_robot2[5] = -self.obs_robot1[5]
        self.obs_robot2[6] = -self.obs_robot1[6]
        self.obs_robot2[7] = -self.obs_robot1[7]
        self.obs_robot2[8] = -self.obs_robot1[0]
        self.obs_robot2[9] = -self.obs_robot1[1]
        self.obs_robot2[10] = -self.obs_robot1[2]
        self.obs_robot2[11] = -self.obs_robot1[3]

        return self.obs_robot1, self.obs_robot2

class Get_modelstate(Node):

    def __init__(self):
        super().__init__('get_modelstate')
        self.subscription = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.listener_callback,
            10)
        self.subscription

    def listener_callback(self, data):
        global observation

        unit_sphere_id = data.name.index('unit_sphere')

        observation[2] = data.pose[unit_sphere_id].position.x
        observation[3] = data.pose[unit_sphere_id].position.y
        observation[4] = data.pose[unit_sphere_id].position.z

class Get_robot_pos(Node):

    def __init__(self):
        super().__init__('robot1_head_pos')
        self.subscription = self.create_subscription(
            LinkStates,
            '/gazebo/link_states',
            self.listener_callback,
            10)
        self.subscription

    def listener_callback(self, data):
        global observation

        robot_1_head_link_id = data.name.index('robot_1::robot_1_body_link')
        observation[0] = data.pose[robot_1_head_link_id].position.x
        observation[1] = data.pose[robot_1_head_link_id].position.y
        robot_2_head_link_id = data.name.index('robot_2::robot_2_body_link')
        observation[5] = data.pose[robot_2_head_link_id].position.x
        observation[6] = data.pose[robot_2_head_link_id].position.y

if __name__ == '__main__':
    rclpy.init(args=None)
    
    gz_env = GazeboEnv()
    get_modelstate = Get_modelstate()
    get_robot_pos = Get_robot_pos()

    ####### initialize environment hyperparameters ######
    env_name = "self_play"

    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 3000                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(5000)         # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################
   
    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    # state space dimension
    state_dim = 12
    action_dim = 5

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    gz_env.get_logger().info("current logging run number for " + str(env_name) + " : " + str(run_num))
    gz_env.get_logger().info("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0 

    directory = "PPO_model"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)

    #####################################################

    if random_seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    gz_env.get_logger().info("Started training at (GMT) : " + str(start_time))
    gz_env.get_logger().info("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0    

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(gz_env)
    executor.add_node(get_modelstate)
    executor.add_node(get_robot_pos)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = gz_env.create_rate(2)

    try:
        while rclpy.ok():
            while time_step <= max_training_timesteps:

                gz_env.get_logger().info('Initializing')
                state_robot1, state_robot2 = gz_env.reset()
                current_ep_reward = 0

                for t in range(1, max_ep_len + 1):
                    # select action with policy
                    action_robot1 = ppo_agent.select_action(state_robot1)
                    action_robot2_, _, _ = ppo_agent.policy_old.act(torch.FloatTensor(state_robot2).to(device))
                    action_robot2 = action_robot2_.item()
                    #obs_robot1, obs_robot2, reward, done
                    state_robot1, state_robot2, reward, done = gz_env.step(action_robot1, action_robot2)

                    # saving reward and is_terminals
                    ppo_agent.buffer.rewards.append(reward)
                    ppo_agent.buffer.is_terminals.append(done)

                    time_step +=1
                    current_ep_reward += reward

                    # update PPO agent
                    if time_step % update_timestep == 0:
                        ppo_agent.update()

                    # if continuous action space; then decay action std of output action distribution
                    if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                        ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

                    # log in logging file
                    if time_step % log_freq == 0:

                        # log average reward till last episode
                        log_avg_reward = log_running_reward / log_running_episodes
                        log_avg_reward = round(log_avg_reward, 4)

                        log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                        log_f.flush()

                        log_running_reward = 0
                        log_running_episodes = 0

                    # printing average reward
                    if time_step % print_freq == 0:

                        # print average reward till last episode
                        print_avg_reward = print_running_reward / print_running_episodes
                        print_avg_reward = round(print_avg_reward, 2)

                        gz_env.get_logger().info("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                        print_running_reward = 0
                        print_running_episodes = 0

                    # save model weights
                    if time_step % save_model_freq == 0:
                        checkpoint_path = directory + "PPO_{}_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained, time_step)
                        gz_env.get_logger().info("--------------------------------------------------------------------------------------------")
                        gz_env.get_logger().info("saving model at : " + checkpoint_path)
                        ppo_agent.save(checkpoint_path)
                        gz_env.get_logger().info("model saved")
                        gz_env.get_logger().info("Elapsed Time  : " + str(datetime.now().replace(microsecond=0) - start_time))
                        gz_env.get_logger().info("--------------------------------------------------------------------------------------------")

                    # break; if the episode is over
                    if done:
                        break

                print_running_reward += current_ep_reward
                print_running_episodes += 1

                log_running_reward += current_ep_reward
                log_running_episodes += 1

                i_episode += 1

            log_f.close()

            # print total training time
            gz_env.get_logger().info("============================================================================================")
            end_time = datetime.now().replace(microsecond=0)
            gz_env.get_logger().info("Started training at (GMT) : " + str(start_time))
            gz_env.get_logger().info("Finished training at (GMT) : " + str(end_time))
            gz_env.get_logger().info("Total training time  : " + str(end_time - start_time))
            gz_env.get_logger().info("============================================================================================")

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    rclpy.shutdown()
    #executor_thread.join()
