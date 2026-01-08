import numpy as np
import cv2, time
from os import path
from matplotlib import pyplot as plt

import gym
from gym import spaces
from gym.utils import seeding

from generator.envs.simulator.Slam_Segmentation_Simulator import SS_Simulator
import random



class RobotExplorationT0(gym.Env):
    def __init__(self, config_path='config_exploration_segmentation_local.yaml'):
        # print("arg(config_path):", config_path)
        if config_path.startswith("/"):
            fullpath = config_path
        else:
            fullpath = path.join(path.dirname(__file__), "config", config_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        print("arg(config_path):", fullpath)

        print('\033[1;31;47m Config: {}\033[0m'.format(config_path))

        print('\033[1;31;47m Room_Segmentation_Generation \033[0m')
        self.env_now_num = 0
        self.sim = SS_Simulator(fullpath, RL_flag=True)
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.last_map = self.sim.get_state()
        self.last_action = None



        self.sequence_info = [0,0,0,0]
        self.restart_flag = False
        self.done_flag = False 
        self.sucess_flag = False
        self.last_pixel = 0
        self.ratio_set_len = 500
        self.ratio_temp_set_len = 20
        self.delta_ratio_set = [-1 for i in range(self.ratio_set_len)]
        self.delta_ratio_temp_set = []
        self.restart_info = ""
        self.test_map_num = len(self.sim.map_id_set)

        self.env_states_schedule = 0

        # self.statistic_max_reward = 0
        # self.statistic_min_reward = float('inf')
        self.env_now_num = self.env_now_num + 1
        self.check_structure = True

        self.poses_history = []
        self.poses_times = []

        self.turn_times = 0
        self.turn_times_one = 0
        self.stop_times = 0
        self.last_turn = ''
        self.first = False
        self.first_step = False
    

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        # print("arg(mode):", mode)
        return

        


    def reset(self, order=True):
        # print("arg(order):", order)
        self.restart_flag = False
        self.done_flag = False
        self.sucess_flag = False
        self.delta_ratio_temp_set = []
        self.delta_ratio_set = [-1 for i in range(self.ratio_set_len)]
        if self.first_step == True:
            print("\033[34m Exploration completed\033[0m")
        print("\n***********************")
        print("\033[31m Reset NEW ENV !\033[0m")
        reset_start_time = time.time()
        reset_ll_time = time.time()
        if self.first_step == True :
            print("\033[92m Rendering Datasets\033[0m")
            self.sim.slam_map_contour_all_room()
            reset_end_time = time.time()
            reset_elapsed_time = reset_end_time - reset_start_time
            reset_elapsed_time2 = reset_end_time - reset_ll_time
            print("Time-RESET-dataset: {:.2f}s-{:.2f}s".format(reset_elapsed_time, reset_elapsed_time2))
            reset_ll_time = time.time()
        print("\033[92m Generating Simulated Home\033[0m")
        # self.sim.connect_area_slam()

        self.sim.reset(order=True)

        reset_end_time = time.time()
        reset_elapsed_time = reset_end_time - reset_start_time
        reset_elapsed_time2 = reset_end_time - reset_ll_time
        print("Time-RESET-home: {:.2f}s-{:.2f}s".format(reset_elapsed_time, reset_elapsed_time2))
        reset_ll_time = time.time()

        self.last_map = self.sim.get_state()
        self.last_action = None
        # print("stop_times: ", self.stop_times)
        # print("turn_times: ", self.turn_times)


        self.poses_history = []
        self.poses_times = []
        self.turn_times = 0
        self.turn_times_one = 0
        self.stop_times = 0
        self.last_turn = ''

        # print(self.sequence_info)
        self.env_now_num=self.env_now_num+1
        self.sequence_info[1]=self.sim.map_id
        self.sequence_info[3]="{}/{}={}%".format(self.env_now_num, self.test_map_num, int(1.0*self.env_now_num/self.test_map_num*100))
        if self.first_step == True:
            if self.env_states_schedule == 0:
                print("\033[33m Forced Stop!\033[0m")
            elif self.env_states_schedule == 1:
                print("\033[33m Task successful!\033[0m")
            else:
                print("\033[33m Robot stuck!\033[0m")
            # print("\033[35m :::step number: {}; current id: {}\033[0m".format(self.sequence_info[0], self.sequence_info[1]))


        print(":::step number: ", self.sequence_info[0], "env number: "+self.sequence_info[3], "current id: ", self.sequence_info[1])
        print("\033[31m Reset completed!\033[0m")
        print("***********************\n")
        self.env_states_schedule = 0
        # self.first = True
        print("\033[34m Start exploring home {}  ...\033[0m".format(self.sim.map_id))
        return self._get_obs()







    def step(self, action):
        if self.first_step == False:
            self.first_step = True
            print("first running")
        action = int(action)
        assert action in [0, 1, 2]
        filer_ = int(action)
        action = ['forward', 'left', 'right'][action]
        crush_flag = self.sim.moveRobot(action)


        obs = self._get_obs()


        exit_index = -1
        for index_h, history_nav in enumerate(self.poses_history):
            y_beta = abs(self.sim.robotPose[0] - history_nav[0])
            x_beta = abs(self.sim.robotPose[1] - history_nav[1])
            t_beta = abs(self.sim.robotPose[2] - history_nav[2])
            if (y_beta < 2 and x_beta < 2) and t_beta < 5*np.pi/180:
                self.poses_times[index_h] = self.poses_times[index_h] + 1
                exit_index = index_h
                break
        if exit_index == -1:
            self.poses_history.append(self.sim.robotPose)
            self.poses_times.append(1)

        if crush_flag == True:
            if action == 'forward':
                self.turn_times = 0
            else:
                self.turn_times = self.turn_times + 1
        if exit_index == -1:
            exit_num = 0
        else:
            exit_num =  self.poses_times[exit_index]



        aground_flag = self.sim.aground_test()
        reward = 0 # for model testing 


        a_others = [0,1,2]
        a_others.remove(filer_)

        if crush_flag == False:
            random.shuffle(a_others)
            for i_aa in a_others:
                i_aa = ['forward', 'left', 'right'][i_aa]
                # print("\033[32m Collision compensation: {}\033[0m".format(i_aa))
                crush_flag = self.sim.moveRobot(i_aa)
                obs = self._get_obs()
                if crush_flag == True:
                    # print("\033[32m {} OK\033[0m".format(i_aa))
                    if i_aa == 'forward':
                        self.turn_times = 0
                    else:
                        self.turn_times = self.turn_times + 1
                    break
                else:
                    # print("\033[32m {} Fail\033[0m".format(i_aa))
                    pass
        if self.turn_times > 50:
            _crush_ = self.sim.moveRobot('forward')
            obs = self._get_obs()
            if _crush_ == False:
                for i in range(36):
                    _crush_2_ = self.sim.moveRobot('left')
                    if _crush_2_ == True:
                        self.turn_times = self.turn_times + 1
                    _crush_2_ = self.sim.moveRobot('forward')
                    obs = self._get_obs()
                    if _crush_2_ == True:
                        self.turn_times = 0
                        break
            else:
                self.turn_times = 0
            # print("\033[32m **Compulsory straight: LEFT**\033[0m")
        if exit_num > 15:
            # print("\033[32m Circle compensation:\033[0m")
            exit_num_random_go = np.random.choice([0,1,2])
            crush_flag = self.sim.moveRobot(['forward', 'left', 'right'][exit_num_random_go])
            obs = self._get_obs()
            if crush_flag == True and exit_num_random_go != 0:
                self.turn_times = self.turn_times + 1


        done = (self.sim.measure_ratio() > 0.995)
        if done == True:
            print("\033[31m #Step   ratio tragger of reset\033[0m")
            self.env_states_schedule = 1
        if aground_flag == False:
            done = True
            print("\033[31m #Step   aground tragger of reset\033[0m")
            self.env_states_schedule = 2
        if done == True:
            print("\033[31m Step --> reset\033[0m")
            self.restart_flag = True
        info = {'is_success': done}
        # print(self.sim.measure_ratio())
        self.sequence_info[2]=done
        self.sequence_info[0]=self.sequence_info[0] + 1
        # self._compute_reset_flag()
        

        return obs, reward, done, info


    def close(self):
        pass




    def _get_action_space(self):
        """Forward, left and right"""
        return spaces.Discrete(3)

    def _get_observation_space(self):
        obs = self._get_obs()
        observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')
        return observation_space

    def _get_obs(self):
        """
        """
        observation = self.sim.get_state()
        pose = self.sim.get_pose()

        (rot_y, rot_x) = (int(pose[0]), int(pose[1]))
        rot_theta = -pose[2] * 180. / np.pi + 90  # Upward

        # Pad boundaries
        pad_x, pad_y = int(self.sim.state_size[0]/2. * 1.5), int(self.sim.state_size[1]/2. * 1.5)
        state_size_x, state_size_y = int(self.sim.state_size[0]), int(self.sim.state_size[1])
        if rot_y - pad_y < 0:
            observation = cv2.copyMakeBorder(observation, top=pad_y, bottom=0, left=0, right=0,
                                             borderType=cv2.BORDER_CONSTANT, value=self.sim.map_color['uncertain'])
            rot_y += pad_y
        if rot_x - pad_x < 0:
            observation = cv2.copyMakeBorder(observation, top=0, bottom=0, left=pad_x, right=0,
                                             borderType=cv2.BORDER_CONSTANT, value=self.sim.map_color['uncertain'])
            rot_x += pad_x
        if rot_y + pad_y > observation.shape[0]:
            observation = cv2.copyMakeBorder(observation, top=0, bottom=pad_y, left=0, right=0,
                                             borderType=cv2.BORDER_CONSTANT, value=self.sim.map_color['uncertain'])
        if rot_x + pad_x > observation.shape[1]:
            observation = cv2.copyMakeBorder(observation, top=0, bottom=0, left=0, right=pad_x,
                                             borderType=cv2.BORDER_CONSTANT, value=self.sim.map_color['uncertain'])

        # Rotate global map and crop the local observation
        local_map = observation[rot_y - pad_y:rot_y + pad_y, rot_x - pad_x:rot_x + pad_x]
        M = cv2.getRotationMatrix2D((pad_y, pad_x), rot_theta, 1)
        dst = cv2.warpAffine(local_map, M, (pad_y*2, pad_x*2), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=self.sim.map_color['uncertain'])
        dst = dst[pad_y - int(state_size_y/2.):pad_y + int(state_size_y/2.),
                  pad_x - int(state_size_x/2.):pad_x + int(state_size_x/2.)]
        dst = dst[:,:,np.newaxis]

        # Draw the robot at the center
        cv2.circle(dst, (int(state_size_y/2.), int(state_size_x/2.)), int(self.sim.robotRadius), 50, thickness=-1)
        cv2.rectangle(dst, (int(state_size_y/2.) - int(self.sim.robotRadius),
                            int(state_size_x/2.) - int(self.sim.robotRadius)),
                      (int(state_size_y / 2.) + int(self.sim.robotRadius),
                       int(state_size_x / 2.) + int(self.sim.robotRadius)),
                      50, -1)
        return dst.copy()








