from asyncio import FastChildWatcher
import re
from turtle import begin_fill
from cv2 import FILE_STORAGE_APPEND
import numpy as np
from matplotlib import pyplot as plt
import cv2, os

import yaml
import generator.envs.simulator.util as util
import generator.envs.simulator.segmentation_contour_Reader as contour_Reader
from shapely.geometry import Polygon
from shapely.geometry import Point
import math
import time
import copy
import json
import datetime
import pprint
import random

map_color= {'uncertain':-101, 'free':0, 'obstacle':100}
move_choice= {'forward':np.array([1,0]), 'left': np.array([0,1]), 'right': np.array([0,-1])}

class SS_Simulator():
    def __init__(self, param_file, obstacle_config=None, training = True, order = True, RL_flag = False):
        """ SS_Simulator initilization """
        np.random.seed(0)
        """ define class variable """
        self.m2p= 0
        self.robotRadius= 0
        self.stepLength_linear= 0
        self.stepLength_angular= 0
        self.config_poseInit= np.array([0,0,0])
        self.robotPose_init= np.array([0,0,0])
        self.robotResetRandomPose= 0
        self.obs_num= 0
        self.obs_sizeRange= np.array([0,0])
        self.laser_range=0
        self.laser_fov= 0
        self.laser_resol= 0
        self.laser_noiseSigma= 0
        self.slamErr_linear= 0
        self.slamErr_angular= 0
        self.state_size= np.array([0,0])
        self.world= np.zeros([1,1])
        self.obstacle_config = obstacle_config
        self.traj = []
        self.skip_map_id = []
        self.traverse_end = False
        self.train_flag = training

        self.segmentation_points = []
        self.boundary_points = []
        self.segmentation_points_slam = []

        self.order_id_pointer = 0
        self.down_id = 0
        self.images_save_states = 0
        self.anno_save_states = 0
        self.images_buffer_simple = {}
        self.images_buffer_slam = {}
        self.images_buffer_com = {}

        self.rim_count = 0

        self.scaning_noise_type = 1

        time_run = datetime.datetime.now()

        if time_run.month-10 < 0:
            cc_code = time_run.month + 3
        else:
            cc_code = time_run.month-9


        # self.run_code = str(cc_code) + str(time_run.day) + str(time_run.hour) + str(time_run.minute) + str(time_run.second)


        categories_dataset = [
            {'supercategory': 'none', 'id': 1, 'name': 'room'}
        ]
        # self.sim_dataset_annotions = {'images': [], 'annotations': [], 'categories': categories_dataset}
        # self.sim_dataset_count={'images' : 0, 'annotations' : 0}
        # self.complete_dataset_annotions = {'images': [], 'annotations': [], 'categories': categories_dataset}
        # self.complete_dataset_count={'images' : 0, 'annotations' : 0}
        # self.slam_dataset_annotions = {'images': [], 'annotations': [], 'categories': categories_dataset}
        # self.slam_dataset_count={'images' : 0, 'annotations' : 0}





        self.out_height = 512
        self.out_width = 512
        # self.out_center_height = int(self.out_height//2)
        # self.out_center_width = int(self.out_width//2)

        self.noise_world_flag = False
        self.complete_construciton = True



        self.data_generation_RL_flag = RL_flag




        with open(param_file) as stream:
            self.config = yaml.load(stream)

        if self.config['visualization_enable'] == 1:
            self.dispaly_flag = True
        else:
            self.dispaly_flag = False

        """json reader init"""
        self.json_reader = contour_Reader.contour_Reader(self.config['json_dir'], self.config['meter2pixel'])
        """ set map_color """
        self.map_color = map_color
        if self.config['map_id_set'] == 'all_9999.txt':
            json_name = os.listdir(self.config['json_dir'])
            self.map_id_set = [map_i.split('.')[0] for map_i in json_name]
            self.map_id_set = np.array(self.map_id_set)
        else:
            # self.map_id_set = np.loadtxt(os.path.join(os.path.dirname(__file__), "../", self.config['map_id_set']), str)
            self.map_id_set = np.loadtxt(self.config['map_id_set'], str)
            print("map_id_file: ", self.config['map_id_set'], ", os.join: ", os.path.join(os.path.dirname(__file__), "../", self.config['map_id_set']))
            print("map_id_set: ", len(self.map_id_set))
        """ Initialize user config param """
        self.initialize_param(self.config)

        if os.path.exists(os.path.join(self.config['annotions_save_dir'], self.config['annotions_file_name'])):
            with open(os.path.join(self.config['annotions_save_dir'], self.config['annotions_file_name'])) as json_file:
                self.all_dataset_annotions = json.load(json_file)
            with open(os.path.join(self.config['annotions_save_dir'], self.config['info_file_name'])) as info_file:
                info_data = json.load(info_file)
            self.all_dataset_count={'images' : info_data['images_count'], 'annotations' : info_data['annotations_count']}
            self.json_name_files = info_data['json_lists']
            current_json_lsit = self.config['map_id_set'].split('/')[-1]
            if current_json_lsit in self.json_name_files:
                self.order_id_pointer = info_data["start_order"]
                self.rim_count = info_data["epoch"]
                print('\033[1;31;47m Continue Generation for existing lists: {}\033[0m'.format(current_json_lsit))
            else:
                self.order_id_pointer = 0
                self.rim_count = 0
                self.json_name_files.append(current_json_lsit)
                print('\033[1;31;47m Continue Generation for NEW lists: {}\033[0m'.format(current_json_lsit))
            print("testing json type: ", type(info_data["start_order"]), type(info_data["epoch"]), type(info_data['images_count']), type(info_data['annotations_count']),type(self.all_dataset_annotions['annotations'][0]['segmentation'][0][0]),"<--|-->", info_data["start_order"], info_data["epoch"],info_data['images_count'], info_data['annotations_count'],self.all_dataset_annotions['annotations'][0]['segmentation'][0][0])
        else:
            self.all_dataset_annotions = {'images': [], 'annotations': [], 'categories': categories_dataset}
            self.all_dataset_count={'images' : 0, 'annotations' : 0}
            self.order_id_pointer = 0 #normal
            self.rim_count = 0 #normal
            self.json_name_files = []
            self.json_name_files.append(self.config['map_id_set'].split('/')[-1])
            print('\033[1;31;47m NEW Generation: {}\033[0m'.format(self.config['map_id_set'].split('/')[-1]))
            print("testing order number: ", type(self.order_id_pointer), type(self.rim_count), self.order_id_pointer, self.rim_count)

        self.run_code = str(cc_code) + str(time_run.day) + str(time_run.hour) + str(time_run.minute)+ 'R{}'.format(self.rim_count)
        print('\033[1;31;47m Time Code: {}\033[0m'.format(self.run_code))
        print(self.config['annotions_file_name'])

        """ set motion choice """
        self.motionChoice= move_choice

        """ pre calculate radius and angle vector that will be used in building map """
        radius_vect= np.arange(self.laser_range+1)
        self.radius_vect= radius_vect.reshape(1, radius_vect.shape[0]) # generate radius vector of [0,1,2,...,laser_range]

        angles_vect = np.arange(-self.laser_fov*0.5, self.laser_fov*0.5,step=self.laser_resol)
        self.angles_vect = angles_vect.reshape(angles_vect.shape[0], 1) # generate angles vector from -laser_angle/2 to laser_angle

        print("\033[32m max_number:{}|min_size:{}|max_size{}|resolution:{}|laser range:{}|laser fov:{}|laser resol:{}|slam error:{},{}|state size:{},{}\
              \033[0m".format(self.obs_num, self.obs_sizeRange[0], self.obs_sizeRange[1], self.m2p, self.laser_range, self.laser_fov, self.laser_resol, self.slamErr_angular, self.slamErr_linear, self.state_size[0], self.state_size[1]))


        self.robotPose = self.robotPose_init
        self.reset(order)

        return

    def initialize_param(self, config):
        """ world & robot param """
        self.m2p= config["meter2pixel"] # X pixel= 1 meter
        self.robotRadius= util.meter2pixel(config["robotRadius"], self.m2p) # robot radius in pixel
        self.stepLength_linear= util.meter2pixel(config["stepLength"]["linear"], self.m2p) # robot linear movement in each step in pixel
        self.stepLength_angular= util.deg2rad( config["stepLength"]["angular"] ) # robot angular movement in each step in rad

        """ robot starting pose """
        # robot starting pose in world coordinate and rad with the form of [y;x;theta]
        self.config_poseInit[0]= config["startPose"]["y"]
        self.config_poseInit[1]= config["startPose"]["x"]
        self.config_poseInit[2]= util.deg2rad( config["startPose"]["theta"] )

        # flag of robot randomly reset start pose in each reset
        self.robotResetRandomPose= config["resetRandomPose"]


        """ obstacle param """
        self.obs_num= config["obstacle"]["number"] # number of obstacle added to the world
        # size of obstacle added in the form of [min;max] in pixel
        self.obs_sizeRange= np.zeros((2,1))
        self.obs_sizeRange[0]= util.meter2pixel( config["obstacle"]["size"]["min"], self.m2p)
        self.obs_sizeRange[1]= util.meter2pixel( config["obstacle"]["size"]["max"], self.m2p)

        """ laser param """
        self.laser_range= util.meter2pixel(config["laser"]["range"], self.m2p) # laser range in pixel
        self.laser_fov= util.deg2rad( config["laser"]["fov"] ) # laser field of view in rad
        self.laser_resol= util.deg2rad( config["laser"]["resolution"] ) # laser rotation resolution in rad
        self.laser_noiseSigma= config["laser"]["noiseSigma"] # sigma of Gaussian distribution of laser noise

        """ slam error """
        self.slamErr_linear= config["slamError"]["linear"] # slam linear registration error in pixel?
        self.slamErr_angular= util.deg2rad( config["slamError"]["angular"] ) # slam rotational registration error in rad

        """ state size """
        self.state_size= ( config["stateSize"]["x"] * self.m2p, config["stateSize"]["y"] * self.m2p ) # state size in the form of [x;y]

        """ unknown mode """
        self.is_exploration = (config["mode"] == 0)


        return


    def map_render(self, pic):
        new_color_pic = np.ones([pic.shape[0], pic.shape[1],3],dtype=np.uint8)*220
        color_nn = 0
        (h0,w0) = pic.shape[:2]
        for i in range(h0):
            for j in range(w0):
                if pic[i][j] == self.map_color['obstacle']:
                    new_color_pic[i][j] = [0,0,0]
                if pic[i][j] == self.map_color['free']:
                    new_color_pic[i][j] = [255,255,255]
        return new_color_pic
        # cv2.destroyAllWindows()
        # cv2.imwrite("/media/sxy/home_HDD/project/python/HouseExpo-master/png/sxy" + "/" +str(self.down_id)+ '{}'.format(name)+'_'+'.png', new_color_pic)



    def show_contour_pic(self, pic, points, name='contour', suspension=True, stop=True, expand=False):
        if expand == True:
            flag_0, pic_ta, points_ta = self.target_image_generation(pic, points)
            if flag_0 == True:
                pic = pic_ta
                points = points_ta
        new_color_pic = np.ones([pic.shape[0], pic.shape[1],3],dtype=np.uint8)*220
        color_all = [[255,0,0],[0,255,0], [0,0,255],[72,61,139],[0,191,255],[70,130,180],[176,196,222],[34,139,34],
                     [107,142,35],[50,205,50],[255,255,0],[255,215,0],[189,183,107],[0, 255, 255],[255, 0, 255],[128, 128, 128]]
        color_nn = 0
        (h0,w0) = pic.shape[:2]
        if len(pic.shape) == 2:
            for i in range(h0):
                for j in range(w0):
                    if pic[i][j] == self.map_color['obstacle']:
                        new_color_pic[i][j] = (0,0,0)
                    if pic[i][j] == self.map_color['free']:
                        new_color_pic[i][j] = (255,255,255)
        else:
            for i in range(h0):
                for j in range(w0):
                    new_color_pic[i][j] = pic[i][j]
        for a_iii in points:
            color_cc = color_all[color_nn]
            for p_iii in range(len(a_iii)):
                if p_iii == (len(a_iii) - 1):
                    cv2.line(new_color_pic, a_iii[p_iii], a_iii[0],color_cc,2)
                else:
                    # print((a_iii[p_iii],a_iii[p_iii + 1]))
                    cv2.line(new_color_pic, a_iii[p_iii], a_iii[p_iii + 1],color_cc,2)
            color_nn = color_nn + 1
            if color_nn == len(color_all):
                color_nn = 0
        cv2.imshow(name, new_color_pic)
        if stop == False:
            cv2.waitKey(100)
        else:
            cv2.waitKey(0)
        if suspension == False:
            cv2.destroyWindow(name)

        # cv2.destroyAllWindows()
        # cv2.imwrite("/media/sxy/home_HDD/project/python/HouseExpo-master/png/sxy" + "/" +str(self.down_id)+ '{}'.format(name)+'_'+'.png', new_color_pic)
        self.down_id = self.down_id + 1

    def search_black_conotur_pixel_100(self, image):
        re = []
        for raw in range(len(image)):
            for i in range(len(image[raw])):
                if image[raw][i] == self.map_color['obstacle']:
                    t = image[max(raw-1,0):min(raw+2, len(image)),max(i-1,0):min(i+2,len(image[raw]))]
                    if (t == self.map_color['free']).any():
                        re.append([int(i),int(raw)])
        return re


    def slam_map_contour_one_room_test_information(self, slam_m, room_conto):
        Binary_slam_m = np.zeros_like(slam_m,np.uint8)
        room_free_pic = np.zeros_like(slam_m,np.uint8)
        # Binary_slam_m[slam_m == self.map_color['obstacle']] = self.map_color['obstacle']
        # Binary_slam_m[slam_m == self.map_color['uncertain']] = self.map_color['obstacle']
        Binary_slam_m[slam_m == self.map_color['free']] = 255
        if (room_conto[0][0]==room_conto[-1][0]) and (room_conto[0][1]==room_conto[-1][1]):
            cv2.fillPoly(room_free_pic, [np.array(room_conto[:-1],np.int32)], 255)
        else:
            cv2.fillPoly(room_free_pic, [np.array(room_conto,np.int32)], 255)
        room_free_pic[Binary_slam_m == 0] = 0
        cv2.imshow('test_contour', room_free_pic)
        cv2.waitKey(0)
        contours_slam, hierarchy_slam = cv2.findContours(room_free_pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours_slam))
        # color_all = [[255,0,0],[0,255,0], [0,0,255],[72,61,139],[0,191,255],[70,130,180],[176,196,222],[34,139,34],
        #              [107,142,35],[50,205,50],[255,255,0],[255,215,0],[189,183,107],[0, 255, 255],[255, 0, 255],[128, 128, 128]]
        # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        # print(len(contours_slam),"dsaf")
        re_con = []
        if len(contours_slam) > 1:
            print(contours_slam)
        if len(contours_slam) == 0:
            return re_con 
        else:
            contours_slam = contours_slam[0]
            if (contours_slam[0][0][0] == contours_slam[-1][0][0]) and (contours_slam[0][0][1] == contours_slam[-1][0][1]):
                contours_slam = contours_slam[:-1]
            for i_re_con in contours_slam:
                re_con.append(i_re_con[0])
            return re_con
    def slam_map_contour_all_room_test_information(self):
        self.segmentation_points_slam = []
        for index_test, points_test_segmentation in enumerate(self.segmentation_points):
            print("\nroom ",index_test, ":")
            self.segmentation_points_slam.append(self.slam_map_contour_one_room_test_information(self.slamMap, points_test_segmentation))
        
        self.show_contour_pic(self.slamMap, self.segmentation_points_slam, 'slam',stop=False)


    def slam_map_contour_one_room(self, slam_m, room_conto):
        def search_black_conotur_pixel_draw(image):
            tu = []
            for raw in range(len(image)):
                for i in range(len(image[raw])):
                    if image[raw][i] == 0:
                        t = image[max(raw-1,0):min(raw+2, len(image)),max(i-1,0):min(i+2,len(image[raw]))]
                        if (t == 255).any():
                            tu.append([raw,i])
            for tu_i in tu:
                image[tu_i[0]][tu_i[1]] = 255
        Binary_slam_m = np.zeros_like(slam_m,np.uint8)
        room_free_pic = np.zeros_like(slam_m,np.uint8)

        Binary_slam_m[slam_m == self.map_color['free']] = 255
        # if (room_conto[0][0]==room_conto[-1][0]) and (room_conto[0][1]==room_conto[-1][1]):
        #     print("room_conto [0] = room_conto [-1]")
            # cv2.fillPoly(room_free_pic, [np.array(room_conto[:-1],np.int32)], 255)
        # else:
        #     cv2.fillPoly(room_free_pic, [np.array(room_conto,np.int32)], 255)


        cv2.fillPoly(room_free_pic, [np.array(room_conto,np.int32)], 255)
        room_free_pic[Binary_slam_m == 0] = 0
        # cv2.imshow('contour0000', room_free_pic)
        # cv2.waitKey(0)
        search_black_conotur_pixel_draw(room_free_pic)
        # cv2.imshow('contour2222', room_free_pic)
        # cv2.waitKey(0)
        # contours_slam, hierarchy_slam = cv2.findContours(room_free_pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_slam, hierarchy_slam = cv2.findContours(room_free_pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # color_all = [[255,0,0],[0,255,0], [0,0,255],[72,61,139],[0,191,255],[70,130,180],[176,196,222],[34,139,34],
        #              [107,142,35],[50,205,50],[255,255,0],[255,215,0],[189,183,107],[0, 255, 255],[255, 0, 255],[128, 128, 128]]
        # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        # print(len(contours_slam),"dsaf")

       
        re_con = []
        if len(contours_slam) == 0:
            re_con = []
        elif len(contours_slam) == 1:
            # print("len(contours_slam) == 1")
            contours_slam = contours_slam[0]
            # if (contours_slam[0][0][0] == contours_slam[-1][0][0]) and (contours_slam[0][0][1] == contours_slam[-1][0][1]):
            #     contours_slam = contours_slam[:-1]
            for i_re_con in contours_slam:
                re_con.append([int(i_re_con[0][0]), int(i_re_con[0][1])])
                # if slam_m[int(i_re_con[0][1])][int(i_re_con[0][0])] == self.map_color['free']:
                #     pass
                # else:
                #     re_con.append([int(i_re_con[0][0]), int(i_re_con[0][1])])
        else:
            # max_points_num = 0
            max_points_area_ = 0
            index_points_num = 0
            for switch_con in contours_slam:
                re_con_ppp = []
                for i_re_con in switch_con:
                    re_con_ppp.append([int(i_re_con[0][0]), int(i_re_con[0][1])])
                    # if slam_m[int(i_re_con[0][1])][int(i_re_con[0][0])] == self.map_color['free']:
                    #     pass
                    # else:
                    #     re_con_ppp.append([int(i_re_con[0][0]), int(i_re_con[0][1])])
                re_con.append(re_con_ppp)
            # print("re_con: ", re_con)
            for i_slam_cc in range(len(re_con)):
                if len(re_con[i_slam_cc]) < 3:
                    __a__ = 0
                else:
                    __a__ = Polygon(re_con[i_slam_cc]).area
                if max_points_area_ < __a__:
                    max_points_area_ = __a__
                    index_points_num = i_slam_cc
                # if len(contours_slam[i_slam_cc]) > max_points_num:
                #     index_points_num = i_slam_cc
                #     max_points_num = len(contours_slam[i_slam_cc])
            # print("   index: ",index_points_num)
            # print("contours_slam: ", len(contours_slam[index_points_num]), "re_con: ", len(re_con[index_points_num]))
            if max_points_area_ > 400:
                re_con = re_con[index_points_num]
                # if (re_con[0][0] == re_con[-1][0]) and (re_con[0][1] == re_con[-1][1]):
                #     re_con = re_con[:-1]
            else:
                re_con = []

        change_ttt = []
        if re_con != []:
            convex_temp = np.zeros_like(self.world,np.uint8)
            convex_temp_2 = np.zeros_like(self.world,np.uint8)
            # origin_s = Polygon(room_conto)
            shrink_s = Polygon(re_con)
            if shrink_s.area > 400:
                shrink_convex = shrink_s.convex_hull
                bbox = shrink_convex.bounds
                min_x_shrink,min_y_shrink,max_x_shrink,max_y_shrink = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
                for i in range(min_x_shrink, max_x_shrink + 1):
                    for j in range(min_y_shrink, max_y_shrink + 1):
                        p_lot = Point(i,j)
                        if(p_lot.within(shrink_convex)==True or p_lot.within(shrink_convex.boundary)==True):
                            convex_temp[j][i] = 255
                cv2.fillPoly(convex_temp_2, [np.array(room_conto,np.int32)],255)
                convex_temp[convex_temp_2 == 0] = 0
                # contours_change, hierarchy_change = cv2.findContours(convex_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours_change, hierarchy_change = cv2.findContours(convex_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # print("contours_change: ", len(contours_change))
                
                contours_change = contours_change[0]
                
                for A3_p in contours_change:
                    change_ttt.append([int(A3_p[0][0]), int(A3_p[0][1])])
                    # A3_p = A3_p[0]
                    # if slam_m[int(A3_p[1])][int(A3_p[0])] != self.map_color['free']:
                    #     change_ttt.append([int(A3_p[0]), int(A3_p[1])])
                if (change_ttt[0][0] != change_ttt[-1][0]) or (change_ttt[0][1] != change_ttt[-1][1]):
                    change_ttt.append(change_ttt[0])
                if (re_con[0][0] != re_con[-1][0]) or (re_con[0][1] != re_con[-1][1]):
                    re_con.append(re_con[0])
            else:
                re_con = []
        return re_con, change_ttt

    def points_conversion_simMap(self):
        points_conversion_results = []
        for points_room_con in self.segmentation_points:
            points_conversion_one_room=[]
            area_simple = np.zeros_like(self.simple_pic,np.uint8)
            cv2.fillPoly(area_simple, [np.array(points_room_con,np.int32)],255)
            # simple_contours, hierarchy_change = cv2.findContours(area_simple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            simple_contours, hierarchy_change = cv2.findContours(area_simple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # print("simple_contours: ", len(simple_contours))
            simple_contours = simple_contours[0]
            for Sim_p in simple_contours:
                points_conversion_one_room.append([int(Sim_p[0][0]), int(Sim_p[0][1])])
            if (points_conversion_one_room[0][0] != points_conversion_one_room[-1][0]) or (points_conversion_one_room[0][1] != points_conversion_one_room[-1][1]):
                points_conversion_one_room.append(points_conversion_one_room[0])
            points_conversion_results.append(points_conversion_one_room)
        return points_conversion_results


    def slam_map_contour_all_room(self):
        self.segmentation_points_slam = []
        self.segmentation_points_slam_convex = []
        for index_test, points_test_segmentation in enumerate(self.segmentation_points):
            contour_a, contour_b = self.slam_map_contour_one_room(self.slamMap, points_test_segmentation)
            if len(contour_a) > 2 and len(contour_b) > 2:
                self.segmentation_points_slam.append(contour_a)
                self.segmentation_points_slam_convex.append(contour_b)
        # if self.data_generation_RL_flag == False:
        #     self.show_contour_pic(self.slamMap, self.segmentation_points_slam, 'slam',stop=False)
        #     self.show_contour_pic(self.slamMap, [], 'map_origin',stop=False)
        #     self.show_contour_pic(self.slamMap, self.segmentation_points_slam_convex, 'slam2',stop=False)
        #     self.show_contour_pic(self.slamMap, self.segmentation_points, 'slam3',stop=False)
        #     self.show_contour_pic(self.slamMap, self.segmentation_points_slam, 'dataset_slam',stop=False, expand=True)
        # else:
        #     self.show_contour_pic(self.slamMap, self.segmentation_points_slam_convex, 'dataset_slam',stop=False, expand=True)
        if self.noise_world_flag == False:
            slamMap_noise = np.ones(self.slamMap.shape)*self.map_color["uncertain"]
            slamMap_noise[self.slamMap == self.map_color['obstacle']] = self.map_color['obstacle']
            slamMap_noise[self.slamMap == self.map_color['free']] = self.map_color['free']
            # slamMap_noise[self.slamMap == self.map_color['uncertain']] = self.map_color['uncertain']
            laser_all = []
            for index_new, points_new_segmentation in enumerate(self.segmentation_points_slam):
                if len(points_new_segmentation) > 1:
                    free_space_obstacles_2 = Polygon(points_new_segmentation)
                    bbox = free_space_obstacles_2.bounds
                    min_x,min_y,max_x,max_y = bbox[0],bbox[1],bbox[2],bbox[3]
                    self.laser_contour_noise(slamMap_noise, min_y, max_y, min_x, max_x, points_new_segmentation, laser_all)
            self.slamMap = slamMap_noise




        self.segmentation_points_slam_com = []
        self.segmentation_points_slam_convex_com = []
        for index_test, points_test_segmentation in enumerate(self.segmentation_points):
            contour_a_com, contour_b_com = self.slam_map_contour_one_room(self.comMap, points_test_segmentation)
            if len(contour_a_com) > 2 and len(contour_b_com) > 2:
                self.segmentation_points_slam_com.append(contour_a_com)
                self.segmentation_points_slam_convex_com.append(contour_b_com)

        if self.noise_world_flag == False:
            comMap_noise = np.ones(self.comMap.shape)*self.map_color["uncertain"]
            comMap_noise[self.comMap == self.map_color['obstacle']] = self.map_color['obstacle']
            comMap_noise[self.comMap == self.map_color['free']] = self.map_color['free']
            laser_all = []
            for index_new, points_new_segmentation_com in enumerate(self.segmentation_points_slam_com):
                if len(points_new_segmentation_com) > 1:
                    free_space_obstacles_2 = Polygon(points_new_segmentation_com)
                    bbox = free_space_obstacles_2.bounds
                    min_x,min_y,max_x,max_y = bbox[0],bbox[1],bbox[2],bbox[3]
                    self.laser_contour_noise(comMap_noise, min_y, max_y, min_x, max_x, points_new_segmentation_com, laser_all)
            self.comMap = comMap_noise
            # print("noise_world_flag = False")



        if self.noise_world_flag == True and self.rim_count == 0:
            self.segmentation_points_simple = self.points_conversion_simMap()




        # # annotions
        # if self.data_generation_RL_flag == True:
        #     self.show_contour_pic(self.simple_pic, self.segmentation_points, 'simple room',stop=False)
        flag_trans_2, images_result_slam, contour_result_slam = self.target_image_generation(self.slamMap, self.segmentation_points_slam_convex)
        flag_trans_3, images_result_com, contour_result_com = self.target_image_generation(self.comMap, self.segmentation_points_slam_convex_com)
        if self.noise_world_flag == True and self.rim_count == 0:
            flag_trans_1, images_result_simple, contour_result_simple = self.target_image_generation(self.simple_pic, self.segmentation_points_simple)
            if flag_trans_1 == False:
                print('\033[1;31;47m Data Conversion Error \033[0m')
                return
        if flag_trans_2 == False or flag_trans_3 == False:
            print('\033[1;31;47m Data Conversion Error \033[0m')
            return
        if self.scaning_noise_type == 1:
            scaning_code_str = "_g"
        else:
            scaning_code_str = "_n"
        if self.noise_world_flag == True:
            slam_pic_name = "sl_"+self.run_code+scaning_code_str+"_"+self.map_id+".png"
            com_pic_name = "co_"+self.run_code+scaning_code_str+"_"+self.map_id+".png"
            simple_pic_name = "simple_"+self.map_id+".png"
        else:
            slam_pic_name = "psl_"+self.run_code+scaning_code_str+"_"+self.map_id+".png"
            com_pic_name = "pco_"+self.run_code+scaning_code_str+"_"+self.map_id+".png"

        if self.rim_count > 0:
            rotation = np.random.choice([0,1],p=[0.8,0.2])
            if rotation == 1:
                flag_rot_slam, rot_image_slam, trans_M_slam, rot_contours_slam = self.Map_Rotation(images_result_slam, contour_result_slam)
                if flag_rot_slam == True:
                    images_result_slam = rot_image_slam
                    contour_result_slam = rot_contours_slam
                flag_rot_com, rot_image_com, trans_M_com, rot_contours_com = self.Map_Rotation(images_result_com, contour_result_com)
                if flag_rot_com == True:
                    images_result_com = rot_image_com
                    contour_result_com = rot_contours_com
        # self.training_data_generation(self.segmentation_points, self.map_id+"_complete_"+self.run_code+".png", self.complete_dataset_annotions['annotations'], self.complete_dataset_annotions['images'], self.complete_dataset_count['images'], self.complete_dataset_count['annotations'])
        # print("ssssss: result", contour_result_slam, '\nslam convex', self.segmentation_points_slam_convex, '\n slam', self.segmentation_points_slam)
        self.training_data_generation(contour_result_slam, slam_pic_name)
        self.training_data_generation(contour_result_com, com_pic_name)
        if self.noise_world_flag == True and self.rim_count == 0:
            self.training_data_generation(contour_result_simple, simple_pic_name)
        images_result_slam = self.map_render(images_result_slam)
        images_result_com = self.map_render(images_result_com)
        if self.noise_world_flag == True and self.rim_count == 0:
            images_result_simple = self.map_render(images_result_simple)
        self.images_buffer_slam.update({slam_pic_name: images_result_slam})
        self.images_buffer_com.update({com_pic_name: images_result_com})
        if self.noise_world_flag == True and self.rim_count == 0:
            self.images_buffer_simple.update({simple_pic_name: images_result_simple})
        self.images_save_states = self.images_save_states + 1
        if self.images_save_states == 3:
            for write_image2 in self.images_buffer_slam:
                cv2.imwrite(self.config['images_save_dir']+'/' + write_image2, self.images_buffer_slam[write_image2])
            for write_image3 in self.images_buffer_com:
                cv2.imwrite(self.config['images_save_dir']+'/' + write_image3, self.images_buffer_com[write_image3])
            for write_image1 in self.images_buffer_simple:
                cv2.imwrite(self.config['images_save_dir']+'/' + write_image1, self.images_buffer_simple[write_image1])
            print('\033[1;31;47m New Maps were saved~\033[0m')
            self.images_buffer_simple = {}
            self.images_buffer_slam = {}
            self.images_buffer_com = {}
            self.images_save_states = 0
            self.save_annotations(self.all_dataset_annotions, self.config['annotions_save_dir'])
            print('\033[1;31;47m Annotation Json is saved~\033[0m')



        if self.dispaly_flag == True:
            if self.data_generation_RL_flag == False:
                self.show_contour_pic(self.comMap, [], 'complete slam map',stop=False)
                self.show_contour_pic(self.comMap, self.segmentation_points_slam_com, 'complete contours',stop=False)
                self.show_contour_pic(images_result_com, contour_result_com, 'expand complete slam contours',stop=False)
            else:
                self.show_contour_pic(images_result_com, contour_result_com, 'expand complete slam contours',stop=False)
        if self.dispaly_flag == True:
            if self.data_generation_RL_flag == False:
                self.show_contour_pic(self.slamMap, [], 'noise slam map',stop=False)
                self.show_contour_pic(self.slamMap, self.segmentation_points_slam, 'noise slam contours',stop=False)
                self.show_contour_pic(images_result_slam, contour_result_slam, 'expand noise slam contours',stop=False)
            else:
                self.show_contour_pic(images_result_slam, contour_result_slam, 'expand noise slam contours',stop=False)
        if self.noise_world_flag == True and self.rim_count == 0:
            if self.dispaly_flag == True:
                if self.data_generation_RL_flag == False:
                    self.show_contour_pic(self.simple_pic, [], 'simple slam map',stop=False)
                    self.show_contour_pic(images_result_simple, contour_result_simple, 'simple contours',stop=False)
                else:
                    self.show_contour_pic(images_result_simple, contour_result_simple, 'simple slam contours',stop=False)
        length_of_contours = []
        # length_of_contours.append([len(lo) for lo in self.segmentation_points_slam_convex])
        # length_of_contours.append([len(lo) for lo in self.segmentation_points_slam_convex_com])
        # if self.noise_world_flag == True and self.rim_count == 0:
        #     length_of_contours.append([len(lo) for lo in self.segmentation_points_simple])
        length_of_contours.append([len(lo) for lo in contour_result_slam])
        length_of_contours.append([len(lo) for lo in contour_result_com])
        if self.noise_world_flag == True and self.rim_count == 0:
            length_of_contours.append([len(lo) for lo in contour_result_simple])
        print("map: ",self.map_id, " ;epoch: ",self.rim_count, " ;order: ",self.order_id_pointer, " ;length of the annotion: ",length_of_contours)
        if self.noise_world_flag == True and self.rim_count == 0:
            self.Save_logs_png(images_result_slam, images_result_com, contour_result_slam, contour_result_com, images_result_simple, contour_result_simple)
        else:
            self.Save_logs_png(images_result_slam, images_result_com, contour_result_slam, contour_result_com)
        


    def Map_Rotation(self, map_image_o, contours_o):
        border = 6
        backcolor = self.map_color["uncertain"]
        obscolor = self.map_color["obstacle"]
        freecolor = self.map_color["free"]
        d2 = map_image_o != backcolor
        # print(d2)
        arg_map = np.argwhere(d2)
        max_y = np.max(arg_map[:,0])
        max_x=np.max(arg_map[:,1])
        min_y = np.min(arg_map[:,0])
        min_x = np.min(arg_map[:,1])
        if (max_y - min_y) > (map_image_o.shape[0] - 2*border) or (max_x - min_x) > (map_image_o.shape[1] - 2*border):
            return False, [],[],[]
        diagonal = math.sqrt((max_y - min_y) * (max_y - min_y) + (max_x - min_x) * (max_x - min_x))
        if diagonal > (min(map_image_o.shape[1], map_image_o.shape[0]) - 2*border):
            return False, [],[],[]
        border = 5
        center_x = (max_x - min_x)/2.0 + min_x
        center_y = (max_y - min_y)/2.0 + min_y
        center = np.array([center_x, center_y], dtype=np.float64)
        rot = np.random.randint(360)
        M_All = cv2.getRotationMatrix2D(center,rot,1)
        a_ro = cv2.warpAffine(map_image_o,M_All,(map_image_o.shape[1],map_image_o.shape[0]),flags=cv2.INTER_NEAREST,borderValue=backcolor)
        d2 = a_ro != backcolor
        # print(d2)
        arg_map = np.argwhere(d2)
        # print(arg_map)
        max_y = np.max(arg_map[:,0])
        max_x=np.max(arg_map[:,1])
        min_y = np.min(arg_map[:,0])
        min_x = np.min(arg_map[:,1])
        new_center_x = (map_image_o.shape[1]-1)/2.0
        new_center_y = (map_image_o.shape[0]-1)/2.0
        ori_center_x = (max_x - min_x)/2.0 + min_x
        ori_center_y = (max_y - min_y)/2.0 + min_y
        d_x = new_center_x - ori_center_x
        d_y = new_center_y - ori_center_y
        M_t_map = np.array([[1, 0, d_x], [0, 1, d_y]],dtype=np.float64)
        M_All[0][2] = M_All[0][2] + M_t_map[0][2]
        M_All[1][2] = M_All[1][2] + M_t_map[1][2]
        a_input = cv2.warpAffine(map_image_o,M_All,(map_image_o.shape[1],map_image_o.shape[0]),flags=cv2.INTER_NEAREST,borderValue=backcolor)
        c1 = a_input == backcolor
        c2 = a_input == obscolor
        c3 = a_input == freecolor
        if np.all(np.logical_or(np.logical_or(c1,c2),c3)):
            contours_input = []
            for room_con_o in contours_o:
                room_input = []
                convex_te = np.zeros_like(map_image_o,dtype=np.uint8)
                cv2.fillPoly(convex_te, [np.array(room_con_o,dtype=np.int32)],255)
                room_pixel_level, hierarchy_change = cv2.findContours(convex_te, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                room_pixel_level = room_pixel_level[0]
                room_pixel_level = room_pixel_level[:,0]
                tmp_new = np.dot(room_pixel_level, M_All[:, :2].T) + M_All[:, 2]
                tmp_new = np.around(tmp_new)
                tmp_new = np.array(tmp_new,dtype=np.int32)
                # tmp_new[tmp_new[:,0] < 0, 0] = 0
                # tmp_new[tmp_new[:,0] >= map_image_o.shape[1], 0] = map_image_o.shape[1] - 1 
                # tmp_new[tmp_new[:,1] < 0, 1] = 0
                # tmp_new[tmp_new[:,1] >= map_image_o.shape[0], 1] = map_image_o.shape[0] - 1 
                convex_te2 = np.zeros_like(map_image_o,dtype=np.uint8)
                cv2.fillPoly(convex_te2, [tmp_new],255)
                room_approx, hierarchy_change = cv2.findContours(convex_te2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                room_approx = room_approx[0]
                for point_room_new in room_approx:
                    room_input.append([int(point_room_new[0][0]), int(point_room_new[0][1])])
                if (room_input[0][0] != room_input[-1][0]) or (room_input[0][1] != room_input[-1][1]):
                    room_input.append(room_input[0])
                contours_input.append(room_input)
        else:
            return False, [],[],[]

        return True, a_input, M_All, contours_input



    def Map_Rotation_old(self, map_image_o, contours_o, flag_type):
        border = 6
        backcolor = self.map_color["uncertain"]
        obscolor = self.map_color["obstacle"]
        freecolor = self.map_color["free"]
        d2 = map_image_o != backcolor
        # print(d2)
        arg_map = np.argwhere(d2)
        max_y = np.max(arg_map[:,0])
        max_x=np.max(arg_map[:,1])
        min_y = np.min(arg_map[:,0])
        min_x = np.min(arg_map[:,1])
        if (max_y - min_y) >= (map_image_o.shape[0] - 2*border) and (max_x - min_x) >= (map_image_o.shape[1] - 2*border):
            return False, 
        diagonal = math.sqrt((max_y - min_y) * (max_y - min_y) + (max_x - min_x) * (max_x - min_x))
        if diagonal >= (min(map_image_o.shape[1], map_image_o.shape[0]) - 2*border):
            return False,
        border = 5
        if flag_type == 0:
            x_reduce = -min_x + border
            x_add = map_image_o.shape[1] - max_x - border
            if x_reduce > 0:
                x_reduce = 0
            if x_add < 0:
                x_add = 0
            y_reduce = -min_y + border
            y_add = map_image_o.shape[0] - max_y - border
            if y_reduce >0:
                y_reduce = 0
            if y_add < 0:
                y_add = 0
            if x_reduce == x_add:
                trans_x_random1 = 0
            else:
                trans_x_random1 = np.random.randint(x_reduce, x_add)
            if y_add == y_reduce:
                trans_y_random1 = 0
            else:
                trans_y_random1 = np.random.randint(y_reduce, y_add)
            M_All = np.array([[1, 0, trans_x_random1], [0, 1, trans_y_random1]],dtype=np.float64)
            a_input = cv2.warpAffine(map_image_o,M_All,(map_image_o.shape[1],map_image_o.shape[0]),flags=cv2.INTER_NEAREST,borderValue=backcolor)
        else:
            new_center_x = (map_image_o.shape[1]-1)/2.0
            new_center_y = (map_image_o.shape[0]-1)/2.0
            ori_center_x = (max_x - min_x)/2.0 + min_x
            ori_center_y = (max_y - min_y)/2.0 + min_y
            d_x = new_center_x - ori_center_x
            d_y = new_center_y - ori_center_y
            M_x_S = np.array([[1, 0, d_x], [0, 1, d_y]],dtype=np.float64)
            center = np.array([new_center_x, new_center_y], dtype=np.float64)
            rot = np.random.randint(360)
            M_rt = cv2.getRotationMatrix2D(center,rot,1)
            M_All = M_rt.copy()
            b_trt = np.dot(M_x_S[:,2],M_rt[:, :2].T)
            M_All[0][2] = M_All[0][2] + b_trt[0]
            M_All[1][2] = M_All[1][2] + b_trt[1]
            a_ro = cv2.warpAffine(map_image_o,M_All,(map_image_o.shape[1],map_image_o.shape[0]),flags=cv2.INTER_NEAREST,borderValue=backcolor)
            d2 = a_ro != backcolor
            # print(d2)
            arg_map = np.argwhere(d2)
            # print(arg_map)
            max_y = np.max(arg_map[:,0])
            max_x=np.max(arg_map[:,1])
            min_y = np.min(arg_map[:,0])
            min_x = np.min(arg_map[:,1])
            x_reduce = -min_x + border
            x_add = map_image_o.shape[1] - max_x - border
            if x_reduce > 0:
                x_reduce = 0
            if x_add < 0:
                x_add = 0
            y_reduce = -min_y + border
            y_add = map_image_o.shape[0] - max_y - border
            if y_reduce >0:
                y_reduce = 0
            if y_add < 0:
                y_add = 0
            if x_reduce == x_add:
                trans_x_random = 0
            else:
                trans_x_random = np.random.randint(x_reduce, x_add)
            if y_add == y_reduce:
                trans_y_random = 0
            else:
                trans_y_random = np.random.randint(y_reduce, y_add)
            M_t_map = np.array([[1, 0, trans_x_random], [0, 1, trans_y_random]],dtype=np.float64)
            M_All[0][2] = M_All[0][2] + M_t_map[0][2]
            M_All[1][2] = M_All[1][2] + M_t_map[1][2]
            a_input = cv2.warpAffine(map_image_o,M_All,(map_image_o.shape[1],map_image_o.shape[0]),flags=cv2.INTER_NEAREST,borderValue=backcolor)
        c1 = a_input == backcolor
        c2 = a_input == obscolor
        c3 = a_input == freecolor
        if np.all(np.logical_or(np.logical_or(c1,c2),c3)):
            contours_input = []
            for room_con_o in contours_o:
                room_input = []
                convex_te = np.zeros_like(map_image_o,dtype=np.uint8)
                cv2.fillPoly(convex_te, [np.array(room_con_o,dtype=np.int32)],255)
                room_pixel_level, hierarchy_change = cv2.findContours(convex_te, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                room_pixel_level = room_pixel_level[0]
                room_pixel_level = room_pixel_level[:,0]
                tmp_new = np.dot(room_pixel_level, M_All[:, :2].T) + M_All[:, 2]
                tmp_new = np.around(tmp_new)
                tmp_new = np.array(tmp_new,dtype=np.int32)
                tmp_new[tmp_new[:,0] < 0, 0] = 0
                tmp_new[tmp_new[:,0] >= map_image_o.shape[1], 0] = map_image_o.shape[1] - 1 
                tmp_new[tmp_new[:,1] < 0, 1] = 0
                tmp_new[tmp_new[:,1] >= map_image_o.shape[0], 1] = map_image_o.shape[0] - 1 
                convex_te2 = np.zeros_like(map_image_o,dtype=np.uint8)
                cv2.fillPoly(convex_te2, [tmp_new],255)
                room_approx, hierarchy_change = cv2.findContours(convex_te2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                room_approx = room_approx[0]
                for point_room_new in room_approx:
                    room_input.append([int(point_room_new[0][0]), int(point_room_new[0][1])])
                if (room_input[0][0] != room_input[-1][0]) or (room_input[0][1] != room_input[-1][1]):
                    room_input.append(room_input[0])
                contours_input.append(room_input)
        else:
            return False, 

        return a_input, M_All, contours_input



    def target_image_generation(self, map_image,contours, other_grid_color=[]):
        # target_map = np.full((self.out_height,self.out_width,3),color_uncertain, dtype=np.uint8)
        target_map = np.ones([self.out_height,self.out_width])*self.map_color["uncertain"]
        height, width = map_image.shape[0], map_image.shape[1]
        target_contours = []
        # center_height = int(map_image.shape[0]//2)
        # center_width = int(map_image.shape[1]//2)
        if (height<=self.out_height) and (width <= self.out_width):
            offset_height = int((self.out_height - height) // 2)
            offset_width = int((self.out_width - width)//2)
            if (self.out_height < height + offset_height) or (self.out_width < offset_width + width):
                offset_height = -1
                offset_width = -1
                return False, target_map,target_contours
            else:
                d2 = map_image != self.map_color["uncertain"]
                # print(d2)
                arg_map = np.argwhere(d2)
                max_h = np.max(arg_map[:,0])
                max_w=np.max(arg_map[:,1])
                min_h = np.min(arg_map[:,0])
                min_w = np.min(arg_map[:,1])
                diff_h = int(round((height-1-max_h - min_h)/2.0))
                diff_w = int(round((width-1-max_w - min_w)/2.0))

                for h in range(height):
                        for w in range(width):
                            if map_image[h][w] == self.map_color["free"] or map_image[h][w] == self.map_color["obstacle"]:
                                target_map[h+offset_height+diff_h][w+offset_width+diff_w]=map_image[h][w]
                            else:
                                if map_image[h][w] in other_grid_color:
                                    target_map[h+offset_height+diff_h][w+offset_width+diff_w]=map_image[h][w]
                for room_contours in contours:
                    temp_con = []
                    for data_con in room_contours:
                        temp_con.append([data_con[0]+offset_width+diff_w, data_con[1]+offset_height+diff_h])
                    target_contours.append(temp_con)
        else:
            offset_height = -1
            offset_width = -1
            return False, target_map,target_contours
        return True, target_map ,target_contours


    def save_annotations(self, ann, path):
        os.system('mkdir -p {}'.format(path))
        instance_path = os.path.join(path, self.config['annotions_file_name'])
        with open(instance_path, 'w') as f:
            json.dump(ann, f)
        dataset_info = {'images_count':self.all_dataset_count['images'], 'annotations_count':self.all_dataset_count['annotations'], 'start_order':self.order_id_pointer, 'epoch':self.rim_count, 'json_lists':self.json_name_files, 'last_map':self.map_id}
        info_path = os.path.join(path, self.config['info_file_name'])
        with open(info_path, 'w') as g:
            json.dump(dataset_info, g)


    def create_world(self, order=True, padding=10):
        """ read maps in order if True, else randomly sample"""
        #TODO: How to set the order, 
        # Decide on several rounds of dataset training based on stopping conditions and needs for intensive learning
        if order:
            map_id = self.map_id_set[self.order_id_pointer]
            self.order_id_pointer = self.order_id_pointer + 1
            if self.order_id_pointer >= len(self.map_id_set):
                self.order_id_pointer = 0
                # map_id = self.map_id_set[self.order_id_pointer]
                # self.order_id_pointer = self.order_id_pointer + 1
                self.rim_count = self.rim_count + 1
                print('\033[1;31;47m New Circle of Dateset !\033[0m')
                time_run = datetime.datetime.now()

                if time_run.month-10 < 0:
                    cc_code = time_run.month + 3
                else:
                    cc_code = time_run.month-9
                self.run_code = str(cc_code) + str(time_run.day) + str(time_run.hour) + str(time_run.minute)+ 'R{}'.format(self.rim_count)
                print('\033[1;31;47m Time Code: {}\033[0m'.format(self.run_code))
        else:
            map_id = np.random.choice(self.map_id_set)
        input_world, segmentation_input,boundary_points, door_input, re_flag= self.json_reader.read_json(map_id)

        print("contour flag: ", re_flag)

        """ process world into simulator compatible map """
        self.world= self._map_process(input_world, padding=padding)
        self.simple_pic=self.simple_image_generation(self.world, boundary_points)

        self.origin_world_floor = np.zeros_like(self.world)
        self.origin_world_floor[self.world==self.map_color['obstacle']] = self.map_color['obstacle']


        self.boundary_points = boundary_points
        self.segmentation_points = segmentation_input #像素坐标(x,y)；与图像矩阵的索引相反
        self.segmentation_black_conotur_pixels = [[]]
        self.segmentation_area_properties = []
        self.door_points = door_input
        for area_points in self.segmentation_points:
           p1 = Polygon(area_points[:-1]).area
           self.segmentation_area_properties.append({"origin_area":p1, "new_area":p1, "perimeter":-1})
        # self.doorway_points = doorway_points

        if self.dispaly_flag == True:
            if self.data_generation_RL_flag == False:
                self.show_contour_pic(self.world, self.segmentation_points, "empty room", False)

        self.segmentation_points_dynamic_shrinking = copy.deepcopy(self.segmentation_points)

        # self.config_poseInit is relative to the center of the map
        # 要注意整个程序的位姿四舍五入，全文用到位姿的地方一定不要出现位置偏差
        (h,w)= self.world.shape
        self.robotPose_init[0:2]= util.world2mapCoord(self.config_poseInit, (h*0.5,w*0.5), self.m2p)
        self.robotPose_init[2]= self.config_poseInit[2]
        # self.add_obstacle()


        # # 测试边界点的值
        # print("self.map_color['obstacle']: ",self.map_color['obstacle'])
        # print("边界点的值测试点一：",self.world[self.segmentation_points[0][0][1]][self.segmentation_points[0][0][0]],self.world[self.segmentation_points[1][5][1]][self.segmentation_points[1][5][0]])      


        obstacle_result = self.add_obstacle_close_wall()

        if obstacle_result == False:
            re_flag = False



        # # 测试边界点的值
        # print("self.map_color['obstacle']: ",self.map_color['obstacle'])
        # print("边界点的值测试点二：",self.world[self.segmentation_points_dynamic_shrinking[0][6][1]][self.segmentation_points_dynamic_shrinking[0][6][0]],self.world[self.segmentation_points_dynamic_shrinking[1][10][1]][self.segmentation_points_dynamic_shrinking[1][10][0]])
        # # self.add_obstacle()
 
        # print("map_id",map_id, "map_id tyep",type(map_id))
        self.map_id = map_id.copy()
        

        return re_flag

    def _map_process(self,input_world, padding=5):
        """ process input map into simulator compatible map """

        map_gt = np.zeros_like(input_world)

        """ input world obstacle= 0 & free space= 255 | convert into simulator config """
        map_gt[input_world == 0] = self.map_color['obstacle']
        map_gt[input_world == 255] = self.map_color['free']

        # print("map_gt.shape: ", map_gt.shape)
        # # Is the padding necessary here?
        # # The following will eventually be deleted to prevent the transformation of coordinate points from being affected
        # """ crop out redundent obstacle region in boundaries """
        # index = np.where(map_gt == self.map_color['obstacle'])
        # [index_row_max, index_row_min, index_col_max, index_col_min] = \
        #     [np.max(index[0]), np.min(index[0]), np.max(index[1]), np.min(index[1])]

        # print("map_gt range: ",[index_row_max, index_row_min, index_col_max, index_col_min])


        # """ pad the map with obstacle on near boundary """
        # map_gt = np.lib.pad(map_gt, padding, mode='constant', constant_values=self.map_color['obstacle'])

        return map_gt




    def random_multi_obstacle_size(self,switch_flag, obs_type_A, obs_sizeRange):
        
        if switch_flag == 0:
            if (obs_type_A == 0) or (obs_type_A == 2):
                if len(obs_sizeRange) == 3:
                    obs_size_type_a = np.random.choice([0,1,2],p=[0.3,0.5,0.2])
                elif len(obs_sizeRange) == 2:
                    obs_size_type_a = np.random.choice([0,1],p=[0.4,0.6])
                else:
                    obs_size_type_a = 0
                    
                obs_size_a = int(np.random.randint(obs_sizeRange[obs_size_type_a][0],obs_sizeRange[obs_size_type_a][1]+1))
                
                if obs_size_type_a == 0:
                    obs_size_type_b = 0
                    obs_size_b = int(np.random.randint(obs_sizeRange[0][0],obs_size_a+1))
                if obs_size_type_a == 1:
                    obs_size_type_b = np.random.choice([0,1],p=[0.4, 0.6])
                    if obs_size_type_b == 1:
                        obs_size_b = int(np.random.randint(obs_sizeRange[1][0],obs_size_a+1))
                    else:
                        obs_size_b = int(np.random.randint(obs_sizeRange[0][0],obs_sizeRange[0][1]+1))
                if obs_size_type_a == 2:
                    obs_size_type_b = np.random.choice([0,1,2],p=[0.3, 0.4, 0.3])
                    if obs_size_type_b == 2:
                        obs_size_b = int(np.random.randint(obs_sizeRange[2][0],obs_size_a+1))
                    else:
                        obs_size_b = int(np.random.randint(obs_sizeRange[obs_size_type_b][0],obs_sizeRange[obs_size_type_b][1]+1))                

                obs_size_a = max(obs_size_a,obs_size_b)
                obs_size_b = min(obs_size_a,obs_size_b)



            if obs_type_A == 1:
                if len(obs_sizeRange) == 3:
                    obs_size_type_a = np.random.choice([0,1,2],p=[0.2,0.45,0.35])
                elif len(obs_sizeRange) == 2:
                    obs_size_type_a = np.random.choice([0,1],p=[0.35,0.65])
                else:
                    obs_size_type_a = 0
                obs_size_a = int(np.random.randint(obs_sizeRange[obs_size_type_a][0],obs_sizeRange[obs_size_type_a][1]+1))
                if obs_size_type_a == 0:
                    obs_size_type_b = 0
                    obs_size_b = int(np.random.randint(obs_sizeRange[0][0],obs_size_a+1))
                if obs_size_type_a == 1:
                    obs_size_type_b = np.random.choice([0,1],p=[0.4, 0.6])
                    if obs_size_type_b == 1:
                        obs_size_b = int(np.random.randint(obs_sizeRange[1][0],obs_size_a+1))
                    else:
                        obs_size_b = int(np.random.randint(obs_sizeRange[0][0],obs_sizeRange[0][1]+1))
                if obs_size_type_a == 2:
                    obs_size_type_b = np.random.choice([0,1,2],p=[0.3, 0.4, 0.3])
                    if obs_size_type_b == 2:
                        obs_size_b = int(np.random.randint(obs_sizeRange[2][0],obs_size_a+1))
                    else:
                        obs_size_b = int(np.random.randint(obs_sizeRange[obs_size_type_b][0],obs_sizeRange[obs_size_type_b][1]+1))   
                obs_size_a = max(obs_size_a,obs_size_b)
                obs_size_b = min(obs_size_a,obs_size_b)


                # if obs_size_type_a == 0:
                #     obs_size_type_b = np.random.choice([0,1,2],p=[0.3, 0.4, 0.3])
                #     if obs_size_type_b == 0:
                #         obs_size_b = int(np.random.randint(obs_sizeRange[obs_size_type_b][0],obs_sizeRange[obs_size_type_b][1]+1))
                #         obs_size_a = max(obs_size_a,obs_size_b)
                #         obs_size_b = min(obs_size_a,obs_size_b)
                #     else:
                #         obs_size_b = int(np.random.randint(obs_sizeRange[obs_size_type_b][0],obs_sizeRange[obs_size_type_b][1]+1))
                # elif obs_size_type_a == 1:
                #     obs_size_type_b = np.random.choice([1,2],p=[0.6, 0.4])
                #     if obs_size_type_b == 1:
                #         obs_size_b = int(np.random.randint(obs_sizeRange[obs_size_type_b][0],obs_sizeRange[obs_size_type_b][1]+1))
                #         obs_size_a = max(obs_size_a,obs_size_b)
                #         obs_size_b = min(obs_size_a,obs_size_b)
                #     else:
                #         obs_size_b = int(np.random.randint(obs_sizeRange[obs_size_type_b][0],obs_sizeRange[obs_size_type_b][1]+1))
                # if obs_size_type_a == 2:
                #     obs_size_b = int(np.random.randint(obs_sizeRange[obs_size_type_a][0],obs_sizeRange[obs_size_type_a][1]+1))
                #     obs_size_a = max(obs_size_a,obs_size_b)
                #     obs_size_b = min(obs_size_a,obs_size_b)
        # print("-边长a的选择范围:",obs_size_type_a,obs_sizeRange[obs_size_type_a])
        # print("---边长a的值:",obs_size_a)
        # print("-边长b的选择范围:",obs_size_type_b,obs_sizeRange[obs_size_type_b])
        # print("---边长b的值:",obs_size_b)
        return obs_size_a, obs_size_b

    def is_in_line(self, first_x,first_y,second_x,second_y,t_x,t_y,width_2=2):
        A = first_y - second_y
        B = first_x-second_x
        if (B == 0 and A == 0):
            return False
        if B == 0:
            if (abs(t_x-first_x) <= width_2) and (t_y <= (max(first_y,second_y)+width_2)) and (t_y >= (min(first_y,second_y)-width_2)):
                return True
            else:
                return False
        elif A == 0:
            if (abs(t_y-first_y) <= width_2) and (t_x <= (max(first_x,second_x)+width_2)) and (t_x >= (min(first_x,second_x)-width_2)):
                return True
            else:
                return False
        else:
            k = A/B
            b = first_y - k * first_x
            if (t_y <= (max(first_y,second_y)+width_2)) and (t_y >= (min(first_y,second_y)-width_2)) and (t_x <= (max(first_x,second_x)+width_2)) and (t_x >= (min(first_x,second_x)-width_2)):
                A_ = second_y - first_y
                B_ = first_x-second_x
                C_ = (first_y - second_y) * first_x + \
                    (second_x - first_x) * first_y
                dis_t = np.abs(A_ * t_x + B_ * t_y + C_) / (np.sqrt(A*A + B*B))
                if dis_t <= width_2:
                    return True
                else:
                    return False
            else:
                return False





    def Rectangle_generation(self, first_x,first_y,second_x,second_y,le):
        def distance(x1,y1,x2,y2):
            math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

        def find_point(k0,b0,p_x,py, d0, dire):
            c0 = math.sqrt(1.0/(k0*k0+1))
            p2_x_1 = p_x + d0*c0*dire
            p2_y_1 = k0*p2_x_1+b0
            return (int(round(p2_x_1)),int(round(p2_y_1)))

        A = first_y - second_y
        B = first_x-second_x
        if (B == 0 and A == 0):
            return (((first_x,first_y),(second_x,second_y),(first_x,first_y),(second_x,second_y)),((first_x,first_y),(second_x,second_y),(first_x,first_y),(second_x,second_y)))
        if B == 0:
            return (((first_x,first_y),(second_x,second_y),(second_x-le,second_y),(first_x-le,first_y)),((first_x,first_y),(second_x,second_y),(second_x+le,second_y),(first_x+le,first_y)))
        elif A == 0:
            return (((first_x,first_y),(second_x,second_y),(second_x,second_y-le),(first_x,first_y-le)),((first_x,first_y),(second_x,second_y),(second_x,second_y+le),(first_x,first_y+le)))
        else:
            k = A/B
            k2 = -1.0/k
            b = first_y - k * first_x
            b1 = first_y - k2 * first_x
            b2 = second_y - k2 * second_x
            x_1_p1,y_1_p1 = find_point(k2,b1,first_x,first_y,le,1)
            x_1_p2,y_1_p2 = find_point(k2,b2,second_x,second_y,le,1)

            x_2_p1,y_2_p1 = find_point(k2,b1,first_x,first_y,le,-1)
            x_2_p2,y_2_p2 = find_point(k2,b2,second_x,second_y,le,-1)
            return (((first_x,first_y),(second_x,second_y),(x_1_p2,y_1_p2),(x_1_p1,y_1_p1)),((first_x,first_y),(second_x,second_y),(x_2_p2,y_2_p2),(x_2_p1,y_2_p1)))


    
    def search_145_conotur_pixel(self, image):
        re = []
        for raw in range(len(image)):
            for i in range(len(image[raw])):
                if image[raw][i] == 145:
                    t = image[max(raw-1,0):min(raw+2, len(image)),max(i-1,0):min(i+2,len(image[raw]))]
                    if (t != 145).any():
                        re.append([int(i),int(raw)])
        return re




    def cross_point_function(self,line1, line2):  # 计算交点函数
        x1 = line1[0]  # 取直线1的第一个点坐标
        y1 = line1[1]
        x2 = line1[2]  # 取直线1的第二个点坐标
        y2 = line1[3]
    
        x3 = line2[0]  # 取直线2的第一个点坐标
        y3 = line2[1]
        x4 = line2[2]  # 取直线2的第二个点坐标
        y4 = line2[3]
    
        if x2 - x1 == 0:  # L1 直线斜率不存在
            k1 = None
            b1 = 0
        else:
            k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
            b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    
        if (x4 - x3) == 0:  # L2直线斜率不存在操作
            k2 = None
            b2 = 0
        else:
            k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
            b2 = y3 * 1.0 - x3 * k2 * 1.0
    
        if k1 is None and k2 is None:  # L1与L2直线斜率都不存在，两条直线均与y轴平行
            return -1, -1
        elif k1 is not None and k2 is None:  # 若L2与y轴平行，L1为一般直线，交点横坐标为L2的x坐标
            x = x3
            y = k1 * x3 * 1.0 + b1 * 1.0
        elif k1 is None and k2 is not None:  # 若L1与y轴平行，L2为一般直线，交点横坐标为L1的x坐标
            x = x1
            y = k2 * x1 * 1.0 + b2 * 1.0
        else:  # 两条一般直线
            if k1 == k2:  # 两直线斜率相同
                return -1, -1
            else:  # 两直线不平行，必然存在交点
                x = (b2 - b1) * 1.0 / (k1 - k2)
                y = k1 * x * 1.0 + b1 * 1.0
        return x, y


    def line_range_function(self,x1,y1, x2,y2):  # 计算交点函数

        if x2 - x1 == 0:  # L1 直线斜率不存在
            k1 = None
            b1 = 0
        else:
            k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
            b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    

        return k1, b1

    
    def line_segmentation_length (self,x1,y1, x2,y2,leng):
        k,b = self.line_range_function(x1,y1, x2,y2)
        min_x_p = min(x1,x2)
        max_x_p = max(x1,x2)
        min_y_p = min(y1,y2)
        max_y_p = max(y1,y2)
        if k==None:
            dir = (y2 - y1)/abs(y2 - y1)
            p_y = y1 + dir*leng
            p_x = x1
            # print("n,y2,y1,p_y",n,y2,y1,p_y)
            if (p_y <= max_y_p) and (p_y >= min_y_p):
                return int(round(p_x)), int(round(p_y))
            else:
                print("Out of line segment range!")
                return -100,-100
        else:
            dir = (x2 - x1)/abs(x2 - x1)
            c0 = math.sqrt(1.0/(k*k+1))
            p_x = x1 + leng*c0*dir
            p_y = k*p_x+b
            if (p_y <= max_y_p) and (p_y >= min_y_p):
                if (p_x <= max_x_p) and (p_x >= min_x_p):
                    return (int(round(p_x)),int(round(p_y)))
                else:
                    # print("point_selection_line error")
                    return -100,-100
            else:
                # print("point_selection_line error")
                return -100,-100


    def point_selection_line(self,x1,y1, x2,y2):
        k,b = self.line_range_function(x1,y1, x2,y2)
        ii_point = int(np.random.randint(0,9))
        min_x_p = min(x1,x2)
        max_x_p = max(x1,x2)
        min_y_p = min(y1,y2)
        max_y_p = max(y1,y2)
        if k==None:
            n = (y2 - y1)*1.0/9
            p_y = y1 + n * ii_point
            # print("n,y2,y1,p_y",n,y2,y1,p_y)
            if (p_y <= max_y_p) and (p_y >= min_y_p):
                x1 = int(round(x1)) 
                p_y = int(round(p_y)) 
                return x1, p_y,ii_point
            else:
                # print("point_selection_line error")
                return -100,-100,-100
        else:
            n = (x2 - x1)*1.0/9
            p_x = x1 + n * ii_point
            p_y = k*p_x + b
            # print("n,x2,x1,p_y,p_x,k.b",n,x2,x1,p_y,p_x,k,b)
            if (p_y <= max_y_p) and (p_y >= min_y_p):
                if (p_x <= max_x_p) and (p_x >= min_x_p):
                    p_x = int(round(p_x)) #新加的，来去除显示的时候坐标小数点，不知道会不会产生奇怪的问题
                    p_y = int(round(p_y))
                    return p_x, p_y,ii_point
                else:
                    # print("point_selection_line error")
                    return -100,-100,-100
            else:
                # print("point_selection_line error")
                return -100,-100,-100

    def point_selection_line_v2_parameter(self,x1,y1, x2,y2,point_len=10, starting_point=True, end_point=False, location_ii=-1):
        k,b = self.line_range_function(x1,y1, x2,y2)
        if starting_point == True:
            ii_start = 0
        else:
            ii_start = 1
        if end_point == True:
            ii_end =  point_len
        else:
            ii_end = point_len -1 
        if location_ii == -1:
            ii_point = int(np.random.randint(ii_start,ii_end))
        else:
            ii_point = location_ii
        min_x_p = min(x1,x2)
        max_x_p = max(x1,x2)
        min_y_p = min(y1,y2)
        max_y_p = max(y1,y2)
        if k==None:
            n = (y2 - y1)*1.0/(point_len-1)
            # n = (y2 - y1)*1.0/9
            p_y = y1 + n * ii_point
            # print("n,y2,y1,p_y",n,y2,y1,p_y)
            if (p_y <= max_y_p) and (p_y >= min_y_p):
                x1 = int(round(x1)) 
                p_y = int(round(p_y)) 
                return x1, p_y,ii_point
            else:
                # print("point_selection_line error")
                return -100,-100,-100
        else:
            n = (x2 - x1)*1.0/(point_len-1)
            # n = (x2 - x1)*1.0/9
            p_x = x1 + n * ii_point
            p_y = k*p_x + b
            # print("n,x2,x1,p_y,p_x,k.b",n,x2,x1,p_y,p_x,k,b)
            if (p_y <= max_y_p) and (p_y >= min_y_p):
                if (p_x <= max_x_p) and (p_x >= min_x_p):
                    p_x = int(round(p_x)) #新加的，来去除显示的时候坐标小数点，不知道会不会产生奇怪的问题
                    p_y = int(round(p_y))
                    return p_x, p_y,ii_point
                else:
                    # print("point_selection_line error")
                    return -100,-100,-100
            else:
                # print("point_selection_line error")
                return -100,-100,-100


    def graphic_iscretization(self,type_o, vertex_points,direction_k):
        type_legs = np.random.choice([0,1],p=[0.7,0.3]) #0圆腿, 1方腿
        diameter_list = [1,2,3,4,5]
        re_laser = []
        if type_o == 0:
            # diagonal_len = math.sqrt(((vertex_points[0][0]-vertex_points[2][0]) ** 2 )+ ((vertex_points[0][1]-vertex_points[2][1]) ** 2))
            side_1 = math.sqrt(((vertex_points[0][0]-vertex_points[1][0]) ** 2 )+ ((vertex_points[0][1]-vertex_points[1][1]) ** 2))
            side_2 = math.sqrt(((vertex_points[1][0]-vertex_points[2][0]) ** 2 )+ ((vertex_points[1][1]-vertex_points[2][1]) ** 2))
            if max(side_1,side_2) < 10:
                num_legs = np.random.choice([1,2,3,4],p=[0.2,0.1,0.15,0.55])
            else:
                num_legs = np.random.choice([1,2,3,4],p=[0.05,0.1,0.15,0.7])
            if num_legs == 1:
                one_leg_center_x, one_leg_center_y = self.cross_point_function([vertex_points[0][0],vertex_points[0][1],vertex_points[2][0],vertex_points[2][1]],[vertex_points[1][0],vertex_points[1][1],vertex_points[3][0],vertex_points[3][1]])
                legs_location = (int(round(one_leg_center_x)), int(round(one_leg_center_y)))
                diameter = diameter_list[np.random.choice([0,1,2,3,4],p=[0.3,0.35,0.17,0.1,0.08])]
                if (min(side_1,side_2) > 20) and (diameter <=2):
                    diameter = 3
                while diameter > min(side_1,side_2):
                    diameter = diameter - 1
                if diameter > 0:
                    if diameter == 1:
                        re_laser = [1,[[legs_location,legs_location,legs_location,legs_location]]]
                    else:
                        if type_legs == 0:
                            re_laser = [0,(int(round(diameter*1.0/2)),int(round(diameter*1.0/2))),[legs_location]]
                        else:
                            radius = diameter * 1.0 /2
                            cthe_b = math.cos(np.pi/180* direction_k)
                            sthe_b = math.sin(np.pi/180* direction_k)
                            rect_b_k = [(int(round(legs_location[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(legs_location[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(legs_location[0] + (-radius * cthe_b - radius * sthe_b))), int(round(legs_location[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(legs_location[0] + (radius * cthe_b - radius * sthe_b))), int(round(legs_location[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(legs_location[0] + (radius * cthe_b - -radius * sthe_b))), int(round(legs_location[1] + (radius * sthe_b + -radius * cthe_b))))]
                            re_laser = [1,[rect_b_k]]
                else:
                    re_laser = []
            if num_legs == 2:
                if side_1 > side_2:
                    standard_line = [[vertex_points[1],vertex_points[2]],[vertex_points[3],vertex_points[0]]]
                    _len_ju_ = side_2
                else:
                    _len_ju_ = side_1
                    standard_line = [[vertex_points[0],vertex_points[1]],[vertex_points[2],vertex_points[3]]]
                segment_1_x,segment_1_y, __ = self.point_selection_line_v2_parameter(standard_line[0][0][0],standard_line[0][0][1],standard_line[0][1][0],standard_line[0][1][1],3,True,True,location_ii=1)
                segment_2_x,segment_2_y, __ = self.point_selection_line_v2_parameter(standard_line[1][0][0],standard_line[1][0][1],standard_line[1][1][0],standard_line[1][1][1],3,True,True,location_ii=1)
                diameter = diameter_list[np.random.choice([0,1,2,3,4],p=[0.3,0.35,0.17,0.1,0.08])]
                while ((2*diameter) > math.sqrt(((segment_1_x-segment_2_x) ** 2) + ((segment_1_y - segment_2_y) ** 2))) or (diameter > _len_ju_):
                    diameter = diameter - 1
                if diameter > 0:
                    x_diameter_1, y_diameter_1 = self.line_segmentation_length(segment_1_x,segment_1_y,segment_2_x,segment_2_y,(diameter/2.0))
                    x_diameter_2, y_diameter_2 = self.line_segmentation_length(segment_2_x,segment_2_y,segment_1_x,segment_1_y,(diameter/2.0))
                    x_diameter_center, y_diameter_center, __ = self.point_selection_line_v2_parameter(segment_1_x,segment_1_y,segment_2_x,segment_2_y,3,True,True,location_ii=1)
                    leg_location_1_x, leg_location_1_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_1, y_diameter_1,x_diameter_center, y_diameter_center)
                    leg_location_2_x, leg_location_2_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_2, y_diameter_2,x_diameter_center, y_diameter_center,location_ii=leg_center_ii)
                    leg_location_1 = (int(round(leg_location_1_x)), int(round(leg_location_1_y)))
                    leg_location_2 = (int(round(leg_location_2_x)), int(round(leg_location_2_y)))
                    if diameter == 1:
                        
                        re_laser = [1,[[leg_location_1,leg_location_1,leg_location_1,leg_location_1],[leg_location_2,leg_location_2,leg_location_2,leg_location_2]]]
                    else:
                        if type_legs == 0:
                            re_laser = [0,(int(round(diameter*1.0/2)),int(round(diameter*1.0/2))),[leg_location_1,leg_location_2]]
                        else:
                            radius = diameter * 1.0 /2
                            cthe_b = math.cos(np.pi/180* direction_k)
                            sthe_b = math.sin(np.pi/180* direction_k)
                            rect_b_k_1 = [(int(round(leg_location_1[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_1[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_1[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_1[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_1[1] + (radius * sthe_b + -radius * cthe_b))))]
                            rect_b_k_2 = [(int(round(leg_location_2[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_2[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_2[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_2[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_2[1] + (radius * sthe_b + -radius * cthe_b))))]
                            re_laser = [1,[rect_b_k_1,rect_b_k_2]]
                else:
                    re_laser = []

            if num_legs == 3:
                diameter = diameter_list[np.random.choice([0,1,2,3,4],p=[0.3,0.35,0.2,0.1,0.05])]
                if abs(side_1-side_2) < min((max(side_1,side_2)/4.0),4):
                    segment_1_x,segment_1_y, __ = self.point_selection_line_v2_parameter(vertex_points[1][0],vertex_points[1][1],vertex_points[2][0],vertex_points[2][1],3,True,True,location_ii=1)
                    segment_2_x,segment_2_y, __ = self.point_selection_line_v2_parameter(vertex_points[3][0],vertex_points[3][1],vertex_points[0][0],vertex_points[0][1],3,True,True,location_ii=1)
                    segment_3_x,segment_3_y, __ = self.point_selection_line_v2_parameter(vertex_points[0][0],vertex_points[0][1],vertex_points[1][0],vertex_points[1][1],3,True,True,location_ii=1)
                    segment_4_x,segment_4_y, __ = self.point_selection_line_v2_parameter(vertex_points[2][0],vertex_points[2][1],vertex_points[3][0],vertex_points[3][1],3,True,True,location_ii=1)
                    while (2*diameter) > min(side_1,side_2):
                        diameter = diameter - 1
                    if diameter > 0:
                        x_diameter_1, y_diameter_1 = self.line_segmentation_length(segment_1_x,segment_1_y,segment_2_x,segment_2_y,(diameter/2.0))
                        x_diameter_2, y_diameter_2 = self.line_segmentation_length(segment_2_x,segment_2_y,segment_1_x,segment_1_y,(diameter/2.0))
                        x_diameter_3, y_diameter_3 = self.line_segmentation_length(segment_3_x,segment_3_y,segment_4_x,segment_4_y,(diameter/2.0))
                        x_diameter_4, y_diameter_4 = self.line_segmentation_length(segment_4_x,segment_4_y,segment_3_x,segment_3_y,(diameter/2.0))
                        x_diameter_center, y_diameter_center = self.cross_point_function([segment_1_x,segment_1_y,segment_2_x,segment_2_y],[segment_3_x,segment_3_y,segment_4_x,segment_4_y])
                        if max(side_1,side_2) <= 6:
                            leg_center_location_ii = int(np.random.choice([1,2,3],p=[0.5,0.45,0.05]))
                            leg_center_point_len = 5
                        elif max(side_1,side_2) <= 8:
                            leg_center_location_ii = int(np.random.choice([1,2,3],p=[0.3,0.5,0.2]))
                            leg_center_point_len = 5
                        elif max(side_1,side_2) <= 18:
                            leg_center_location_ii = int(np.random.choice([1,2,3,4,5,6],p=[0.25,0.3,0.2,0.1,0.1,0.05]))
                            leg_center_point_len = 10
                        else:
                            leg_center_location_ii = int(np.random.choice([1,2,3,4,5,6,7,8],p=[0.2,0.2,0.2,0.1,0.1,0.1,0.05,0.05]))
                            leg_center_point_len = 10
                        leg_location_1_x, leg_location_1_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_1, y_diameter_1,x_diameter_center, y_diameter_center,point_len=leg_center_point_len,location_ii=leg_center_location_ii)
                        fixed_distance = math.sqrt(((leg_location_1_x-x_diameter_center) ** 2) + ((leg_location_1_y - y_diameter_center) ** 2))
                        theta_point_1 = math.atan2(leg_location_1_y, leg_location_1_x)
                        theta_point_2 = theta_point_1 + 2*math.pi/3
                        leg_location_2_x = fixed_distance * math.cos(theta_point_2)
                        leg_location_2_y = fixed_distance * math.sin(theta_point_2)
                        theta_point_3 = theta_point_2 + 2*math.pi/3
                        leg_location_3_x = fixed_distance * math.cos(theta_point_3)
                        leg_location_3_y = fixed_distance * math.sin(theta_point_3)
                else:
                    if side_1 > side_2:
                        segment_1_x,segment_1_y, __ = self.point_selection_line_v2_parameter(vertex_points[0][0],vertex_points[0][1],vertex_points[1][0],vertex_points[1][1],3,True,True,location_ii=1)
                        segment_2_x,segment_2_y, __ = self.point_selection_line_v2_parameter(vertex_points[2][0],vertex_points[2][1],vertex_points[3][0],vertex_points[3][1],3,True,True,location_ii=1)
                        segment_3_x,segment_3_y, __ = self.point_selection_line_v2_parameter(vertex_points[1][0],vertex_points[1][1],vertex_points[2][0],vertex_points[2][1],3,True,True,location_ii=1)
                        segment_4_x,segment_4_y, __ = self.point_selection_line_v2_parameter(vertex_points[3][0],vertex_points[3][1],vertex_points[0][0],vertex_points[0][1],3,True,True,location_ii=1)
                        leg_left_point = vertex_points[2]
                        leg_right_point = vertex_points[3]
                    else:
                        segment_1_x,segment_1_y, __ = self.point_selection_line_v2_parameter(vertex_points[1][0],vertex_points[1][1],vertex_points[2][0],vertex_points[2][1],3,True,True,location_ii=1)
                        segment_2_x,segment_2_y, __ = self.point_selection_line_v2_parameter(vertex_points[3][0],vertex_points[3][1],vertex_points[0][0],vertex_points[0][1],3,True,True,location_ii=1)
                        segment_3_x,segment_3_y, __ = self.point_selection_line_v2_parameter(vertex_points[0][0],vertex_points[0][1],vertex_points[1][0],vertex_points[1][1],3,True,True,location_ii=1)
                        segment_4_x,segment_4_y, __ = self.point_selection_line_v2_parameter(vertex_points[2][0],vertex_points[2][1],vertex_points[3][0],vertex_points[3][1],3,True,True,location_ii=1)
                        leg_left_point = vertex_points[0]
                        leg_right_point = vertex_points[3]

                    while (2*diameter) > min(side_1,side_2):
                        diameter = diameter - 1
                    if diameter > 0:
                        x_diameter_1, y_diameter_1 = self.line_segmentation_length(segment_1_x,segment_1_y,segment_2_x,segment_2_y,(diameter/2.0))
                        x_diameter_2, y_diameter_2 = self.line_segmentation_length(segment_2_x,segment_2_y,segment_1_x,segment_1_y,(diameter/2.0))
                        x_diameter_3, y_diameter_3 = self.line_segmentation_length(segment_3_x,segment_3_y,segment_4_x,segment_4_y,(diameter/2.0))
                        x_diameter_4, y_diameter_4 = self.line_segmentation_length(segment_4_x,segment_4_y,segment_3_x,segment_3_y,(diameter/2.0))
                        x_diameter_center, y_diameter_center = self.cross_point_function([segment_1_x,segment_1_y,segment_2_x,segment_2_y],[segment_3_x,segment_3_y,segment_4_x,segment_4_y])

                        if min(side_1,side_2) <= 6:
                            leg_center_location_ii = int(np.random.choice([1,2,3],p=[0.5,0.45,0.05]))
                            leg_center_point_len = 5
                        elif min(side_1,side_2) <= 8:
                            leg_center_location_ii = int(np.random.choice([1,2,3],p=[0.3,0.5,0.2]))
                            leg_center_point_len = 5
                        elif min(side_1,side_2) <= 18:
                            leg_center_location_ii = int(np.random.choice([1,2,3,4,5,6],p=[0.25,0.3,0.2,0.1,0.1,0.05]))
                            leg_center_point_len = 10
                        else:
                            leg_center_location_ii = int(np.random.choice([1,2,3,4,5,6,7,8],p=[0.2,0.2,0.2,0.1,0.1,0.1,0.05,0.05]))
                            leg_center_point_len = 10 
                        leg_location_1_x, leg_location_1_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_1, y_diameter_1,x_diameter_center, y_diameter_center,point_len=leg_center_point_len,location_ii=leg_center_location_ii)
                        leg_location_2_3_c_x, leg_location_2_3_c_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_2, y_diameter_2,x_diameter_center, y_diameter_center,point_len=leg_center_point_len,location_ii=leg_center_location_ii)
                        if max(side_1,side_2) <= 10:
                            leg_center_location_ii_3 = int(np.random.choice([1,2,3,4,5,6,7],p=[0.1,0.15,0.25,0.2,0.15,0.1,0.05]))
                        else:
                            leg_center_location_ii_3 = int(np.random.choice([1,2,3,4,5,6,7],p=[0.2,0.2,0.2,0.15,0.1,0.1,0.05])) 
                        leg_location_2_left_x, leg_location_2_left_y, leg_center_ii = self.point_selection_line_v2_parameter(leg_left_point[0], leg_left_point[1], segment_3_x,segment_3_y,point_len=leg_center_point_len,location_ii=leg_center_location_ii)
                        leg_location_2_right_x, leg_location_2_right_y, leg_center_ii = self.point_selection_line_v2_parameter(leg_right_point[0], leg_right_point[1], segment_4_x,segment_4_y,point_len=leg_center_point_len,location_ii=leg_center_location_ii)

                        leg_location_2_x, leg_location_2_y, leg_center_ii = self.point_selection_line_v2_parameter(leg_location_2_left_x, leg_location_2_left_y, leg_location_2_3_c_x,leg_location_2_3_c_y,point_len=10,location_ii=leg_center_location_ii_3)
                        leg_location_3_x, leg_location_3_y, leg_center_ii = self.point_selection_line_v2_parameter(leg_location_2_right_x, leg_location_2_right_y, leg_location_2_3_c_x,leg_location_2_3_c_y,point_len=10,location_ii=leg_center_location_ii_3)

                if diameter > 0:
                    leg_location_1 = (int(round(leg_location_1_x)), int(round(leg_location_1_y)))
                    leg_location_2 = (int(round(leg_location_2_x)), int(round(leg_location_2_y)))
                    leg_location_3 = (int(round(leg_location_3_x)), int(round(leg_location_3_y)))
                    if diameter == 1:
                        re_laser = [1,[[leg_location_1,leg_location_1,leg_location_1,leg_location_1],[leg_location_2,leg_location_2,leg_location_2,leg_location_2],[leg_location_3,leg_location_3,leg_location_3,leg_location_3]]]
                    else:
                        if type_legs == 0:
                            re_laser = [0,(int(round(diameter*1.0/2)),int(round(diameter*1.0/2))),[leg_location_1,leg_location_2,leg_location_3]]
                        else:
                            radius = diameter * 1.0 /2
                            cthe_b = math.cos(np.pi/180* direction_k)
                            sthe_b = math.sin(np.pi/180* direction_k)
                            rect_b_k_1 = [(int(round(leg_location_1[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_1[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_1[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_1[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_1[1] + (radius * sthe_b + -radius * cthe_b))))]
                            rect_b_k_2 = [(int(round(leg_location_2[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_2[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_2[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_2[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_2[1] + (radius * sthe_b + -radius * cthe_b))))]
                            rect_b_k_3 = [(int(round(leg_location_3[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_3[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_3[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_3[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_3[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_3[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_3[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_3[1] + (radius * sthe_b + -radius * cthe_b))))]
                            re_laser = [1,[rect_b_k_1,rect_b_k_2,rect_b_k_3]]
                else:
                    re_laser = []

            if num_legs == 4:
                diameter = diameter_list[np.random.choice([0,1,2,3,4],p=[0.3,0.35,0.17,0.1,0.08])]
                while (2*diameter) > min(side_1,side_2):
                    diameter = diameter - 1
                if diameter > 0:
                    radius = diameter * 1.0 /2
                    segment_1_x, segment_1_y = self.line_segmentation_length(vertex_points[0][0],vertex_points[0][1],vertex_points[1][0],vertex_points[1][1],(diameter/2.0))
                    segment_2_x, segment_2_y = self.line_segmentation_length(vertex_points[3][0],vertex_points[3][1],vertex_points[2][0],vertex_points[2][1],(diameter/2.0))
                    segment_3_x, segment_3_y = self.line_segmentation_length(vertex_points[0][0],vertex_points[0][1],vertex_points[3][0],vertex_points[3][1],(diameter/2.0))   
                    segment_4_x, segment_4_y = self.line_segmentation_length(vertex_points[1][0],vertex_points[1][1],vertex_points[2][0],vertex_points[2][1],(diameter/2.0))
                    x_diameter_1, y_diameter_1 = self.cross_point_function([segment_1_x,segment_1_y,segment_2_x,segment_2_y],[segment_3_x,segment_3_y,segment_4_x,segment_4_y])

                    segment_5_x, segment_5_y = self.line_segmentation_length(vertex_points[1][0],vertex_points[1][1],vertex_points[2][0],vertex_points[2][1],(diameter/2.0))
                    segment_6_x, segment_6_y = self.line_segmentation_length(vertex_points[0][0],vertex_points[0][1],vertex_points[3][0],vertex_points[3][1],(diameter/2.0))
                    segment_7_x, segment_7_y = self.line_segmentation_length(vertex_points[1][0],vertex_points[1][1],vertex_points[0][0],vertex_points[0][1],(diameter/2.0))
                    segment_8_x, segment_8_y = self.line_segmentation_length(vertex_points[2][0],vertex_points[2][1],vertex_points[3][0],vertex_points[3][1],(diameter/2.0))
                    x_diameter_2, y_diameter_2 = self.cross_point_function([segment_5_x,segment_5_y,segment_6_x,segment_6_y],[segment_7_x,segment_7_y,segment_8_x,segment_8_y])
            
                    segment_9_x, segment_9_y = self.line_segmentation_length(vertex_points[2][0],vertex_points[2][1],vertex_points[1][0],vertex_points[1][1],(diameter/2.0))
                    segment_10_x, segment_10_y = self.line_segmentation_length(vertex_points[3][0],vertex_points[3][1],vertex_points[0][0],vertex_points[0][1],(diameter/2.0))
                    segment_11_x, segment_11_y = self.line_segmentation_length(vertex_points[2][0],vertex_points[2][1],vertex_points[3][0],vertex_points[3][1],(diameter/2.0))   
                    segment_12_x, segment_12_y = self.line_segmentation_length(vertex_points[1][0],vertex_points[1][1],vertex_points[0][0],vertex_points[0][1],(diameter/2.0))
                    x_diameter_3, y_diameter_3 = self.cross_point_function([segment_9_x,segment_9_y,segment_10_x,segment_10_y],[segment_11_x,segment_11_y,segment_12_x,segment_12_y])

                    segment_13_x, segment_13_y = self.line_segmentation_length(vertex_points[3][0],vertex_points[3][1],vertex_points[2][0],vertex_points[2][1],(diameter/2.0))
                    segment_14_x, segment_14_y = self.line_segmentation_length(vertex_points[0][0],vertex_points[0][1],vertex_points[1][0],vertex_points[1][1],(diameter/2.0))
                    segment_15_x, segment_15_y = self.line_segmentation_length(vertex_points[3][0],vertex_points[3][1],vertex_points[0][0],vertex_points[0][1],(diameter/2.0))
                    segment_16_x, segment_16_y = self.line_segmentation_length(vertex_points[2][0],vertex_points[2][1],vertex_points[1][0],vertex_points[1][1],(diameter/2.0))
                    x_diameter_4, y_diameter_4 = self.cross_point_function([segment_13_x,segment_13_y,segment_14_x,segment_14_y],[segment_15_x,segment_15_y,segment_16_x,segment_16_y])
                    
                    x_diameter_center, y_diameter_center = self.cross_point_function([x_diameter_1, y_diameter_1,x_diameter_3, y_diameter_3],[x_diameter_2, y_diameter_2,x_diameter_4, y_diameter_4])
                    
                    if max(side_1,side_2) <= 6: ## max or min?
                        leg_center_location_ii = int(np.random.choice([0,1,2,3],p=[0.25,0.5,0.15,0.1]))
                        leg_center_point_len = 5
                    elif max(side_1,side_2) <= 8:
                        leg_center_location_ii = int(np.random.choice([0,1,2,3],p=[0.15,0.55,0.2,0.1]))
                        leg_center_point_len = 5
                    elif max(side_1,side_2) <= 18:
                        leg_center_location_ii = int(np.random.choice([0,1,2,3,4,5,6,7,8],p=[0.05,0.2,0.3,0.15,0.1,0.05,0.05,0.05,0.05]))
                        leg_center_point_len = 10
                    else:
                        leg_center_location_ii = int(np.random.choice([0,1,2,3,4,5,6,7,8],p=[0.05,0.4,0.2,0.1,0.05,0.05,0.05,0.05,0.05]))
                        leg_center_point_len = 10
                    leg_location_1_x, leg_location_1_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_1, y_diameter_1,x_diameter_center, y_diameter_center,point_len=leg_center_point_len,location_ii=leg_center_location_ii)
                    leg_location_2_x, leg_location_2_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_2, y_diameter_2,x_diameter_center, y_diameter_center,point_len=leg_center_point_len,location_ii=leg_center_location_ii)
                    leg_location_3_x, leg_location_3_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_3, y_diameter_3,x_diameter_center, y_diameter_center,point_len=leg_center_point_len,location_ii=leg_center_location_ii)
                    leg_location_4_x, leg_location_4_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_4, y_diameter_4,x_diameter_center, y_diameter_center,point_len=leg_center_point_len,location_ii=leg_center_location_ii)
                    leg_location_1 = (int(round(leg_location_1_x)), int(round(leg_location_1_y)))
                    leg_location_2 = (int(round(leg_location_2_x)), int(round(leg_location_2_y)))
                    leg_location_3 = (int(round(leg_location_3_x)), int(round(leg_location_3_y)))
                    leg_location_4 = (int(round(leg_location_4_x)), int(round(leg_location_4_y)))  
                    if diameter == 1:
                        
                        re_laser = [1,[[leg_location_1,leg_location_1,leg_location_1,leg_location_1],[leg_location_2,leg_location_2,leg_location_2,leg_location_2],[leg_location_3,leg_location_3,leg_location_3,leg_location_3],[leg_location_4,leg_location_4,leg_location_4,leg_location_4]]]
                    else:
                        if type_legs == 0:
                            re_laser = [0,(int(round(diameter*1.0/2)),int(round(diameter*1.0/2))),[leg_location_1,leg_location_2,leg_location_3,leg_location_4]]
                        else:
                            radius = diameter * 1.0 /2
                            cthe_b = math.cos(np.pi/180* direction_k)
                            sthe_b = math.sin(np.pi/180* direction_k)
                            rect_b_k_1 = [(int(round(leg_location_1[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_1[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_1[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_1[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_1[1] + (radius * sthe_b + -radius * cthe_b))))]
                            rect_b_k_2 = [(int(round(leg_location_2[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_2[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_2[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_2[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_2[1] + (radius * sthe_b + -radius * cthe_b))))]
                            rect_b_k_3 = [(int(round(leg_location_3[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_3[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_3[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_3[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_3[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_3[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_3[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_3[1] + (radius * sthe_b + -radius * cthe_b))))]
                            rect_b_k_4 = [(int(round(leg_location_4[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_4[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_4[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_4[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_4[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_4[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_4[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_4[1] + (radius * sthe_b + -radius * cthe_b))))]
                            re_laser = [1,[rect_b_k_1,rect_b_k_2,rect_b_k_3,rect_b_k_4]]
                else:
                    re_laser = []
        if type_o == 1:
            side_1 = math.sqrt(((vertex_points[4][0][0]-vertex_points[4][1][0]) ** 2 )+ ((vertex_points[4][0][1]-vertex_points[4][1][1]) ** 2))
            side_2 = math.sqrt(((vertex_points[4][1][0]-vertex_points[4][2][0]) ** 2 )+ ((vertex_points[4][1][1]-vertex_points[4][2][1]) ** 2))
            if abs(vertex_points[2]-vertex_points[3]) < (max(vertex_points[2],vertex_points[3])/3.0):
                if abs(vertex_points[2]-vertex_points[3]) < 4:
                    if max(vertex_points[2],vertex_points[3]) <= 8:
                        num_legs = np.random.choice([1,3,4],p=[0.1,0.3,0.6])
                    elif max(vertex_points[2],vertex_points[3]) <= 18:
                        num_legs = np.random.choice([1,3,4],p=[0.1,0.25,0.65])
                    else:
                        num_legs = np.random.choice([1,3,4],p=[0.05,0.25,0.7])
                else:
                    if max(vertex_points[2],vertex_points[3]) <= 8:
                        num_legs = np.random.choice([1,2,3,4],p=[0.2,0.2,0.2,0.4])
                    elif max(vertex_points[2],vertex_points[3]) <= 18:
                        num_legs = np.random.choice([1,2,3,4],p=[0.2,0.15,0.15,0.5])
                    else:
                        num_legs = np.random.choice([1,2,3,4],p=[0.1,0.1,0.2,0.6])
            else:
                if max(vertex_points[2],vertex_points[3]) <= 8:
                    num_legs = np.random.choice([1,2,4],p=[0.2,0.3,0.5])
                elif max(vertex_points[2],vertex_points[3]) <= 18:
                    num_legs = np.random.choice([1,2,3,4],p=[0.15,0.15,0.1,0.6])
                else:
                    num_legs = np.random.choice([1,2,3,4],p=[0.1,0.2,0.1,0.6])
            if num_legs == 1:
                legs_location = (int(round(vertex_points[0])), int(round(vertex_points[1])))
                diameter = diameter_list[np.random.choice([0,1,2,3,4],p=[0.3,0.35,0.17,0.1,0.08])]
                if (min(side_1,side_2) > 20) and (diameter <=2):
                    diameter = 3
                while diameter > min(side_1,side_2):
                    diameter = diameter - 1
                if diameter > 0:
                    if diameter == 1:
                        re_laser = [1,[[legs_location,legs_location,legs_location,legs_location]]]
                    else:
                        re_laser = [0,(int(round(diameter*1.0/2)),int(round(diameter*1.0/2))),[legs_location]]
                else:
                    re_laser = []
            if num_legs == 2:
                if side_1 > side_2:
                    standard_line = [[vertex_points[4][1],vertex_points[4][2]],[vertex_points[4][3],vertex_points[4][0]]]
                    _len_ju_ = side_2
                else:
                    _len_ju_ = side_1
                    standard_line = [[vertex_points[4][0],vertex_points[4][1]],[vertex_points[4][2],vertex_points[4][3]]]
                segment_1_x,segment_1_y, __ = self.point_selection_line_v2_parameter(standard_line[0][0][0],standard_line[0][0][1],standard_line[0][1][0],standard_line[0][1][1],3,True,True,location_ii=1)
                segment_2_x,segment_2_y, __ = self.point_selection_line_v2_parameter(standard_line[1][0][0],standard_line[1][0][1],standard_line[1][1][0],standard_line[1][1][1],3,True,True,location_ii=1)
                diameter = diameter_list[np.random.choice([0,1,2,3,4],p=[0.3,0.35,0.17,0.1,0.08])]
                while ((2*diameter) > math.sqrt(((segment_1_x-segment_2_x) ** 2) + ((segment_1_y - segment_2_y) ** 2))) or (diameter > _len_ju_):
                    diameter = diameter - 1
                if diameter > 0:
                    x_diameter_1, y_diameter_1 = self.line_segmentation_length(segment_1_x,segment_1_y,segment_2_x,segment_2_y,(diameter/2.0))
                    x_diameter_2, y_diameter_2 = self.line_segmentation_length(segment_2_x,segment_2_y,segment_1_x,segment_1_y,(diameter/2.0))
                    x_diameter_center, y_diameter_center, __ = self.point_selection_line_v2_parameter(segment_1_x,segment_1_y,segment_2_x,segment_2_y,3,True,True,location_ii=1)
                    if _len_ju_ < 10:
                        leg_location_1_x, leg_location_1_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_1, y_diameter_1,x_diameter_center, y_diameter_center,3,starting_point=False)
                        leg_location_2_x, leg_location_2_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_2, y_diameter_2,x_diameter_center, y_diameter_center,3,location_ii=leg_center_ii)
                    if _len_ju_ < 20:
                        leg_location_1_x, leg_location_1_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_1, y_diameter_1,x_diameter_center, y_diameter_center,5,starting_point=False)
                        leg_location_2_x, leg_location_2_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_2, y_diameter_2,x_diameter_center, y_diameter_center,5,location_ii=leg_center_ii)
                    else:
                        leg_location_1_x, leg_location_1_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_1, y_diameter_1,x_diameter_center, y_diameter_center,10,starting_point=False)
                        leg_location_2_x, leg_location_2_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_2, y_diameter_2,x_diameter_center, y_diameter_center,10,location_ii=leg_center_ii)
                    leg_location_1 = (int(round(leg_location_1_x)), int(round(leg_location_1_y)))
                    leg_location_2 = (int(round(leg_location_2_x)), int(round(leg_location_2_y)))
                    if diameter == 1:
                        
                        re_laser = [1,[[leg_location_1,leg_location_1,leg_location_1,leg_location_1],[leg_location_2,leg_location_2,leg_location_2,leg_location_2]]]
                    else:
                        if type_legs == 0:
                            re_laser = [0,(int(round(diameter*1.0/2)),int(round(diameter*1.0/2))),[leg_location_1,leg_location_2]]
                        else:
                            radius = diameter * 1.0 /2
                            cthe_b = math.cos(np.pi/180* direction_k)
                            sthe_b = math.sin(np.pi/180* direction_k)
                            rect_b_k_1 = [(int(round(leg_location_1[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_1[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_1[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_1[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_1[1] + (radius * sthe_b + -radius * cthe_b))))]
                            rect_b_k_2 = [(int(round(leg_location_2[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_2[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_2[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_2[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_2[1] + (radius * sthe_b + -radius * cthe_b))))]
                            re_laser = [1,[rect_b_k_1,rect_b_k_2]]
            
                else:
                    re_laser = []
            if num_legs == 3:
                diameter = diameter_list[np.random.choice([0,1,2,3,4],p=[0.3,0.35,0.2,0.1,0.05])]
                if abs(vertex_points[2]-vertex_points[3]) < min((max(vertex_points[2],vertex_points[3])/4.0),4):
                    segment_1_x,segment_1_y, __ = self.point_selection_line_v2_parameter(vertex_points[4][1][0],vertex_points[4][1][1],vertex_points[4][2][0],vertex_points[4][2][1],3,True,True,location_ii=1)
                    segment_2_x,segment_2_y, __ = self.point_selection_line_v2_parameter(vertex_points[4][3][0],vertex_points[4][3][1],vertex_points[4][0][0],vertex_points[4][0][1],3,True,True,location_ii=1)
                    segment_3_x,segment_3_y, __ = self.point_selection_line_v2_parameter(vertex_points[4][0][0],vertex_points[4][0][1],vertex_points[4][1][0],vertex_points[4][1][1],3,True,True,location_ii=1)
                    segment_4_x,segment_4_y, __ = self.point_selection_line_v2_parameter(vertex_points[4][2][0],vertex_points[4][2][1],vertex_points[4][3][0],vertex_points[4][3][1],3,True,True,location_ii=1)
                    while (2*diameter) > min(side_1,side_2):
                        diameter = diameter - 1
                    if diameter > 0:
                        x_diameter_1, y_diameter_1 = self.line_segmentation_length(segment_1_x,segment_1_y,segment_2_x,segment_2_y,(diameter/2.0))
                        x_diameter_2, y_diameter_2 = self.line_segmentation_length(segment_2_x,segment_2_y,segment_1_x,segment_1_y,(diameter/2.0))
                        x_diameter_3, y_diameter_3 = self.line_segmentation_length(segment_3_x,segment_3_y,segment_4_x,segment_4_y,(diameter/2.0))
                        x_diameter_4, y_diameter_4 = self.line_segmentation_length(segment_4_x,segment_4_y,segment_3_x,segment_3_y,(diameter/2.0))
                        x_diameter_center, y_diameter_center = self.cross_point_function([segment_1_x,segment_1_y,segment_2_x,segment_2_y],[segment_3_x,segment_3_y,segment_4_x,segment_4_y])
                        if max(side_1,side_2) <= 6:
                            leg_center_location_ii = int(np.random.choice([1,2],p=[0.6,0.4]))
                            leg_center_point_len = 5
                        elif max(side_1,side_2) <= 8:
                            leg_center_location_ii = int(np.random.choice([1,2,3],p=[0.5,0.3,0.2]))
                            leg_center_point_len = 5
                        elif max(side_1,side_2) <= 18:
                            leg_center_location_ii = int(np.random.choice([1,2,3,4,5,6],p=[0.4,0.2,0.2,0.1,0.05,0.05]))
                            leg_center_point_len = 10
                        else:
                            leg_center_location_ii = int(np.random.choice([1,2,3,4,5,6,7,8],p=[0.2,0.2,0.2,0.1,0.1,0.1,0.05,0.05]))
                            leg_center_point_len = 10
                        leg_location_1_x, leg_location_1_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_1, y_diameter_1,x_diameter_center, y_diameter_center,point_len=leg_center_point_len,location_ii=leg_center_location_ii)
                        fixed_distance = math.sqrt(((leg_location_1_x-x_diameter_center) ** 2) + ((leg_location_1_y - y_diameter_center) ** 2))
                        theta_point_1 = math.atan2(leg_location_1_y, leg_location_1_x)
                        theta_point_2 = theta_point_1 + 2*math.pi/3
                        leg_location_2_x = fixed_distance * math.cos(theta_point_2)
                        leg_location_2_y = fixed_distance * math.sin(theta_point_2)
                        theta_point_3 = theta_point_2 + 2*math.pi/3
                        leg_location_3_x = fixed_distance * math.cos(theta_point_3)
                        leg_location_3_y = fixed_distance * math.sin(theta_point_3)
                else:
                    if side_1 > side_2:
                        segment_1_x,segment_1_y, __ = self.point_selection_line_v2_parameter(vertex_points[4][0][0],vertex_points[4][0][1],vertex_points[4][1][0],vertex_points[4][1][1],3,True,True,location_ii=1)
                        segment_2_x,segment_2_y, __ = self.point_selection_line_v2_parameter(vertex_points[4][2][0],vertex_points[4][2][1],vertex_points[4][3][0],vertex_points[4][3][1],3,True,True,location_ii=1)
                        segment_3_x,segment_3_y, __ = self.point_selection_line_v2_parameter(vertex_points[4][1][0],vertex_points[4][1][1],vertex_points[4][2][0],vertex_points[4][2][1],3,True,True,location_ii=1)
                        segment_4_x,segment_4_y, __ = self.point_selection_line_v2_parameter(vertex_points[4][3][0],vertex_points[4][3][1],vertex_points[4][0][0],vertex_points[4][0][1],3,True,True,location_ii=1)
                        leg_left_point = vertex_points[4][2]
                        leg_right_point = vertex_points[4][3]
                    else:
                        segment_1_x,segment_1_y, __ = self.point_selection_line_v2_parameter(vertex_points[4][1][0],vertex_points[4][1][1],vertex_points[4][2][0],vertex_points[4][2][1],3,True,True,location_ii=1)
                        segment_2_x,segment_2_y, __ = self.point_selection_line_v2_parameter(vertex_points[4][3][0],vertex_points[4][3][1],vertex_points[4][0][0],vertex_points[4][0][1],3,True,True,location_ii=1)
                        segment_3_x,segment_3_y, __ = self.point_selection_line_v2_parameter(vertex_points[4][0][0],vertex_points[4][0][1],vertex_points[4][1][0],vertex_points[4][1][1],3,True,True,location_ii=1)
                        segment_4_x,segment_4_y, __ = self.point_selection_line_v2_parameter(vertex_points[4][2][0],vertex_points[4][2][1],vertex_points[4][3][0],vertex_points[4][3][1],3,True,True,location_ii=1)
                        leg_left_point = vertex_points[4][0]
                        leg_right_point = vertex_points[4][3]

                    while (2*diameter) > min(side_1,side_2):
                        diameter = diameter - 1
                    if diameter > 0:
                        x_diameter_1, y_diameter_1 = self.line_segmentation_length(segment_1_x,segment_1_y,segment_2_x,segment_2_y,(diameter/2.0))
                        x_diameter_2, y_diameter_2 = self.line_segmentation_length(segment_2_x,segment_2_y,segment_1_x,segment_1_y,(diameter/2.0))
                        x_diameter_3, y_diameter_3 = self.line_segmentation_length(segment_3_x,segment_3_y,segment_4_x,segment_4_y,(diameter/2.0))
                        x_diameter_4, y_diameter_4 = self.line_segmentation_length(segment_4_x,segment_4_y,segment_3_x,segment_3_y,(diameter/2.0))
                        x_diameter_center, y_diameter_center = self.cross_point_function([segment_1_x,segment_1_y,segment_2_x,segment_2_y],[segment_3_x,segment_3_y,segment_4_x,segment_4_y])

                        if min(side_1,side_2) <= 6:
                            leg_center_location_ii = int(np.random.choice([1,2],p=[0.6,0.4]))
                            leg_center_point_len = 5
                        elif min(side_1,side_2) <= 8:
                            leg_center_location_ii = int(np.random.choice([1,2,3],p=[0.5,0.3,0.2]))
                            leg_center_point_len = 5
                        elif min(side_1,side_2) <= 18:
                            leg_center_location_ii = int(np.random.choice([1,2,3,4,5],p=[0.4,0.25,0.2,0.1,0.05]))
                            leg_center_point_len = 10
                        else:
                            leg_center_location_ii = int(np.random.choice([1,2,3,4,5,6,7],p=[0.2,0.25,0.2,0.1,0.1,0.1,0.05]))
                            leg_center_point_len = 10

                        if side_1 > side_2:
                            large_1_ = side_1
                            small_2_ = side_2
                        else:
                            large_1_ = side_2
                            small_2_ = side_1                        

                        if large_1_/float(small_2_) < 1.0/3:
                            leg_center_location_ii_2 = leg_center_location_ii
                        elif  large_1_/float(small_2_) < 1.0/2 and float(leg_center_location_ii)/leg_center_point_len <= 1/5:
                            leg_center_location_ii_2 = leg_center_location_ii + 1
                        elif  large_1_/float(small_2_) < 2.0/3 and float(leg_center_location_ii)/leg_center_point_len <= 1/5:
                            leg_center_location_ii_2 = leg_center_location_ii + 2
                        elif  large_1_/float(small_2_) >= 2.0/3 and float(leg_center_location_ii)/leg_center_point_len <= 1/5:
                            leg_center_location_ii_2 = leg_center_location_ii + 3
                        else:
                            leg_center_location_ii_2 = leg_center_location_ii
                        while leg_center_location_ii_2 >= leg_center_point_len:
                            leg_center_location_ii_2 = leg_center_location_ii_2 - 1
                        
                        leg_location_1_x, leg_location_1_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_1, y_diameter_1,x_diameter_center, y_diameter_center,point_len=leg_center_point_len,location_ii=leg_center_location_ii)
                        leg_location_2_3_c_x, leg_location_2_3_c_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_2, y_diameter_2,x_diameter_center, y_diameter_center,point_len=leg_center_point_len,location_ii=leg_center_location_ii_2)
                        leg_center_location_ii_3 = int(np.random.choice([3,4,5,6,7],p=[0.2,0.2,0.2,0.2,0.2]))
                        leg_location_2_left_x, leg_location_2_left_y, leg_center_ii = self.point_selection_line_v2_parameter(leg_left_point[0], leg_left_point[1], segment_3_x,segment_3_y,point_len=leg_center_point_len,location_ii=leg_center_location_ii_2)
                        leg_location_2_right_x, leg_location_2_right_y, leg_center_ii = self.point_selection_line_v2_parameter(leg_right_point[0], leg_right_point[1], segment_4_x,segment_4_y,point_len=leg_center_point_len,location_ii=leg_center_location_ii_2)

                        leg_location_2_x, leg_location_2_y, leg_center_ii = self.point_selection_line_v2_parameter(leg_location_2_left_x, leg_location_2_left_y, leg_location_2_3_c_x,leg_location_2_3_c_y,point_len=10,location_ii=leg_center_location_ii_3)
                        leg_location_3_x, leg_location_3_y, leg_center_ii = self.point_selection_line_v2_parameter(leg_location_2_right_x, leg_location_2_right_y, leg_location_2_3_c_x,leg_location_2_3_c_y,point_len=10,location_ii=leg_center_location_ii_3)
                if diameter > 0:
                    leg_location_1 = (int(round(leg_location_1_x)), int(round(leg_location_1_y)))
                    leg_location_2 = (int(round(leg_location_2_x)), int(round(leg_location_2_y)))
                    leg_location_3 = (int(round(leg_location_3_x)), int(round(leg_location_3_y)))
                    if diameter == 1:
                        re_laser = [1,[[leg_location_1,leg_location_1,leg_location_1,leg_location_1],[leg_location_2,leg_location_2,leg_location_2,leg_location_2],[leg_location_3,leg_location_3,leg_location_3,leg_location_3]]]
                    else:
                        if type_legs == 0:
                            re_laser = [0,(int(round(diameter*1.0/2)),int(round(diameter*1.0/2))),[leg_location_1,leg_location_2,leg_location_3]]
                        else:
                            radius = diameter * 1.0 /2
                            cthe_b = math.cos(np.pi/180* direction_k)
                            sthe_b = math.sin(np.pi/180* direction_k)
                            rect_b_k_1 = [(int(round(leg_location_1[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_1[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_1[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_1[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_1[1] + (radius * sthe_b + -radius * cthe_b))))]
                            rect_b_k_2 = [(int(round(leg_location_2[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_2[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_2[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_2[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_2[1] + (radius * sthe_b + -radius * cthe_b))))]
                            rect_b_k_3 = [(int(round(leg_location_3[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_3[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_3[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_3[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_3[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_3[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_3[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_3[1] + (radius * sthe_b + -radius * cthe_b))))]
                            re_laser = [1,[rect_b_k_1,rect_b_k_2,rect_b_k_3]]
                else:
                    re_laser = []

            if num_legs == 4:
                segment_1_x,segment_1_y, __ = self.point_selection_line_v2_parameter(vertex_points[4][1][0],vertex_points[4][1][1],vertex_points[4][2][0],vertex_points[4][2][1],3,True,True,location_ii=1)
                segment_2_x,segment_2_y, __ = self.point_selection_line_v2_parameter(vertex_points[4][3][0],vertex_points[4][3][1],vertex_points[4][0][0],vertex_points[4][0][1],3,True,True,location_ii=1)
                
                segment_3_x,segment_3_y, __ = self.point_selection_line_v2_parameter(vertex_points[4][0][0],vertex_points[4][0][1],vertex_points[4][1][0],vertex_points[4][1][1],3,True,True,location_ii=1)
                segment_4_x,segment_4_y, __ = self.point_selection_line_v2_parameter(vertex_points[4][2][0],vertex_points[4][2][1],vertex_points[4][3][0],vertex_points[4][3][1],3,True,True,location_ii=1)
                
                diameter = diameter_list[np.random.choice([0,1,2,3,4],p=[0.3,0.35,0.17,0.1,0.08])]
                while (2*diameter) > min(side_1,side_2):
                    diameter = diameter - 1
                if diameter > 0:
                    x_diameter_1, y_diameter_1 = self.line_segmentation_length(segment_1_x,segment_1_y,segment_2_x,segment_2_y,(diameter/2.0))
                    x_diameter_2, y_diameter_2 = self.line_segmentation_length(segment_2_x,segment_2_y,segment_1_x,segment_1_y,(diameter/2.0))
                    x_diameter_3, y_diameter_3 = self.line_segmentation_length(segment_3_x,segment_3_y,segment_4_x,segment_4_y,(diameter/2.0))
                    x_diameter_4, y_diameter_4 = self.line_segmentation_length(segment_4_x,segment_4_y,segment_3_x,segment_3_y,(diameter/2.0))
                    x_diameter_center, y_diameter_center = self.cross_point_function([segment_1_x,segment_1_y,segment_2_x,segment_2_y],[segment_3_x,segment_3_y,segment_4_x,segment_4_y])
                    if side_1 < 10:
                        leg_location_1_x, leg_location_1_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_1, y_diameter_1,x_diameter_center, y_diameter_center,3,starting_point=False)
                        leg_location_2_x, leg_location_2_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_2, y_diameter_2,x_diameter_center, y_diameter_center,3,location_ii=leg_center_ii)
                    if side_1 < 20:
                        leg_location_1_x, leg_location_1_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_1, y_diameter_1,x_diameter_center, y_diameter_center,5,starting_point=False)
                        leg_location_2_x, leg_location_2_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_2, y_diameter_2,x_diameter_center, y_diameter_center,5,location_ii=leg_center_ii)
                    else:
                        leg_location_1_x, leg_location_1_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_1, y_diameter_1,x_diameter_center, y_diameter_center,10,starting_point=False)
                        leg_location_2_x, leg_location_2_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_2, y_diameter_2,x_diameter_center, y_diameter_center,10,location_ii=leg_center_ii)
                    if side_2 < 10:
                        leg_location_3_x, leg_location_3_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_3, y_diameter_3,x_diameter_center, y_diameter_center,3,starting_point=False)
                        leg_location_4_x, leg_location_4_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_4, y_diameter_4,x_diameter_center, y_diameter_center,3,location_ii=leg_center_ii)
                    if side_2 < 20:
                        leg_location_3_x, leg_location_3_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_3, y_diameter_3,x_diameter_center, y_diameter_center,5,starting_point=False)
                        leg_location_4_x, leg_location_4_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_4, y_diameter_4,x_diameter_center, y_diameter_center,5,location_ii=leg_center_ii)
                    else:
                        leg_location_3_x, leg_location_3_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_3, y_diameter_3,x_diameter_center, y_diameter_center,10,starting_point=False)
                        leg_location_4_x, leg_location_4_y, leg_center_ii = self.point_selection_line_v2_parameter(x_diameter_4, y_diameter_4,x_diameter_center, y_diameter_center,10,location_ii=leg_center_ii)
                    leg_location_1 = (int(round(leg_location_1_x)), int(round(leg_location_1_y)))
                    leg_location_2 = (int(round(leg_location_2_x)), int(round(leg_location_2_y)))
                    leg_location_3 = (int(round(leg_location_3_x)), int(round(leg_location_3_y)))
                    leg_location_4 = (int(round(leg_location_4_x)), int(round(leg_location_4_y)))
                    if diameter == 1:
                        re_laser = [1,[[leg_location_1,leg_location_1,leg_location_1,leg_location_1],[leg_location_2,leg_location_2,leg_location_2,leg_location_2],[leg_location_3,leg_location_3,leg_location_3,leg_location_3],[leg_location_4,leg_location_4,leg_location_4,leg_location_4]]]
                    else:
                        if type_legs == 0:
                            re_laser = [0,(int(round(diameter*1.0/2)),int(round(diameter*1.0/2))),[leg_location_1,leg_location_2,leg_location_3,leg_location_4]]
                        else:
                            radius = diameter * 1.0 /2
                            cthe_b = math.cos(np.pi/180* direction_k)
                            sthe_b = math.sin(np.pi/180* direction_k)
                            rect_b_k_1 = [(int(round(leg_location_1[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_1[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_1[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_1[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_1[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_1[1] + (radius * sthe_b + -radius * cthe_b))))]
                            rect_b_k_2 = [(int(round(leg_location_2[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_2[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_2[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_2[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_2[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_2[1] + (radius * sthe_b + -radius * cthe_b))))]
                            rect_b_k_3 = [(int(round(leg_location_3[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_3[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_3[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_3[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_3[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_3[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_3[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_3[1] + (radius * sthe_b + -radius * cthe_b))))]
                            rect_b_k_4 = [(int(round(leg_location_4[0] + (-radius * cthe_b - -radius * sthe_b))), int(round(leg_location_4[1] + (-radius * sthe_b + -radius * cthe_b)))),
                                            (int(round(leg_location_4[0] + (-radius * cthe_b - radius * sthe_b))), int(round(leg_location_4[1] + (-radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_4[0] + (radius * cthe_b - radius * sthe_b))), int(round(leg_location_4[1] + (radius * sthe_b + radius * cthe_b)))),
                                            (int(round(leg_location_4[0] + (radius * cthe_b - -radius * sthe_b))), int(round(leg_location_4[1] + (radius * sthe_b + -radius * cthe_b))))]
                            re_laser = [1,[rect_b_k_1,rect_b_k_2,rect_b_k_3,rect_b_k_4]]
            
                else:
                    re_laser = []
                
        if type_o == 2:
            if len(vertex_points) == 3:
                num_legs = np.random.choice([1,3],p=[0.4,0.6])
            elif len(vertex_points) == 4:
                num_legs = np.random.choice([1,2,3,4],p=[0.1,0.2,0.3,0.4])
            elif len(vertex_points) > 4:
                num_legs = np.random.choice([3,4,5,6],p=[0.2,0.4,0.2,0.2])
            area_type_2 = Polygon(vertex_points)
            area_type_2_bounds = area_type_2.bounds
            min_x_1,min_y_1,max_x_1,max_y_1 = area_type_2_bounds[0],area_type_2_bounds[1],area_type_2_bounds[2],area_type_2_bounds[3]

            if area_type_2.area < (4 * num_legs):
                diameter = 1
            elif area_type_2.area < (9 * num_legs):
                diameter = diameter_list[np.random.choice([0,1],p=[0.7,0.3])]
            elif area_type_2.area < (16 * num_legs):
                diameter = diameter_list[np.random.choice([0,1,2],p=[0.5,0.35,0.15])]
            elif area_type_2.area < (30 * num_legs):
                diameter = diameter_list[np.random.choice([0,1,2,3],p=[0.4,0.4,0.15,0.05])]
            else:
                diameter = diameter_list[np.random.choice([0,1,2,3,4],p=[0.3,0.35,0.17,0.1,0.08])]
            
            obstacle_theta_type_2 = np.random.random()*360
            cthe_b = math.cos(np.pi/180* obstacle_theta_type_2)
            sthe_b = math.sin(np.pi/180* obstacle_theta_type_2)
            re_tmp_type_2 = []
            radius = diameter * 1.0 /2
            for i__ in range(num_legs):
                type_2_count = 0
                type_2_flag_ok = False
                # while (diameter > 0) and (type_2_flag_ok == False):
                while (type_2_count <100) and (type_2_flag_ok == False):
                    type_2_count = type_2_count + 1
                    obs_y_type_2 = int(np.random.randint(min_y_1,max_y_1))
                    obs_x_type_2 = int(np.random.randint(min_x_1,max_x_1))
                    if diameter == 1:
                        if (Point(obs_x_type_2, obs_y_type_2).within(area_type_2) == True):
                            re_tmp_type_2.append([(obs_x_type_2, obs_y_type_2),(obs_x_type_2, obs_y_type_2),(obs_x_type_2, obs_y_type_2),(obs_x_type_2, obs_y_type_2)])
                            type_2_flag_ok = True
                            break
                    else:
                        
                        rect_b = [[int(round(obs_x_type_2 + (-radius * cthe_b - -radius * sthe_b))), int(round(obs_y_type_2 + (-radius * sthe_b + -radius * cthe_b)))],
                                        [int(round(obs_x_type_2 + (-radius * cthe_b - radius * sthe_b))), int(round(obs_y_type_2 + (-radius * sthe_b + radius * cthe_b)))],
                                        [int(round(obs_x_type_2 + (radius * cthe_b - radius * sthe_b))), int(round(obs_y_type_2 + (radius * sthe_b + radius * cthe_b)))],
                                        [int(round(obs_x_type_2 + (radius * cthe_b - -radius * sthe_b))), int(round(obs_y_type_2 + (radius * sthe_b + -radius * cthe_b)))]]
                        area_rand_legs = Polygon(rect_b)
                        area_irregular_obstacle = Polygon(vertex_points)
                        expansive_area_rand_legs = area_rand_legs.buffer(1,join_style=2,cap_style=2)
                        intersection_area_rand_legs = area_rand_legs.intersection(area_irregular_obstacle).area
                        intersection_expansive_area_rand_legs = expansive_area_rand_legs.intersection(area_irregular_obstacle).area
                        if abs(intersection_expansive_area_rand_legs - expansive_area_rand_legs.area) < 0.1:
                            if abs(intersection_area_rand_legs - area_rand_legs.area) < 0.1:
                                for try_obstacle_irregular in re_tmp_type_2:
                                    area_try_obstacle_irregular = Polygon(try_obstacle_irregular)
                                    if area_try_obstacle_irregular.intersection(area_irregular_obstacle).area / area_irregular_obstacle < 0.4:
                                        if type_legs == 1:
                                            re_tmp_type_2.append(rect_b)
                                        else:
                                            re_tmp_type_2.append((obs_x_type_2, obs_y_type_2))
                                        type_2_flag_ok = True
                                        break
                    # if type_2_flag_ok == False:
                    #     diameter = diameter - 1
                if type_2_flag_ok == False:
                    re_tmp_type_2 =[]
                    break
                                        
            if len(re_tmp_type_2) > 0:
                if type_legs == 1 or diameter == 1:
                    re_laser = [1, re_tmp_type_2]
                else:
                    re_laser = [0,(int(round(diameter*1.0/2)),int(round(diameter*1.0/2))),re_tmp_type_2]
            else:
                re_laser = []
                
        return re_laser
            



    # def get_main_directions(self,points_test_segmentation):
    #     re = {}
    #     re_re = (0,0)
    #     for i_g in range(len(points_test_segmentation)-1):
    #         dy = points_test_segmentation[i_g+1][1]-points_test_segmentation[i_g][1]
    #         dx = points_test_segmentation[i_g+1][0]-points_test_segmentation[i_g][0]
    #         if dx == 0:
    #             kt = 90000
    #         else:
    #             kt = int(math.atan(dy*1.0/dx)*1000)
    #         if kt in re:
    #             re[kt] = re[kt] + 1
    #         else:
    #             re.update({kt:1})
    #     flag_add = False
    #     if len(re) > 15:
    #         flag_add = True
    #     degree_out_3 = list(range(0,181,3))
    #     degree_out_3_1 = degree_out_3[:-1]
    #     degree_out_3_2 = degree_out_3[1:]
    #     degree_out_3_3 = list(zip(degree_out_3_1,degree_out_3_2))
    #     degree_out = [0]*len(degree_out_3_3)
    #     if len(re) > 15:
    #         for i in re:
    #             if flag_add == 
            


    # def Point_Amplification(self,point_group,map):
    #     def GeneralEquation(first_x,first_y,second_x,second_y):
    #         A = first_y - second_y
    #         B = first_x - second_x
    #         if B == 0:
    #             return int(1),0,first_x
    #         elif A == 0:
    #             return int(0),0,first_y
    #         else:
    #             K = 1.0*A/B
    #             B = first_y - K * first_x
            
    #         return int(0),K, B
    #     def distance(x1,y1,x2,y2):
    #         return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
        
    #     # print("[......] rendering sampled points")
    #     contour = copy.deepcopy(self.this_black_contour)
    #     # point_group = self.this_json_data[self.this_map_id]['areas']
    #     re_areas = []
    #     for points in point_group:
    #         re_segment = []
    #         # statistic_anno_points_num = 0
    #         #这里的points是带着首尾重复点的
    #         for i in range(len(points)-1):
    #             len_seg = distance(points[i][0],points[i][1],points[i+1][0],points[i+1][1])
    #             # len_seg = int(round(len_seg))
    #             # interval_pixels_len = 10
    #             interval_pixels_len = 2
    #             re_segment.append(points[i]) #import
    #             if len_seg > (1.5 * interval_pixels_len):
    #                 # print("len_seg", len_seg)
    #                 sample_num = int(round(len_seg/interval_pixels_len))
    #                 # sample_num = int(len_seg/interval_pixels_len)
    #                 # print("sample_num", sample_num)
    #                 ty,k,b = GeneralEquation(points[i][0],points[i][1],points[i+1][0],points[i+1][1])
    #                 interval = (points[i+1][ty]-points[i][ty])/float(sample_num)
    #                 if (points[i+1][ty]-points[i][ty]) >= 0:
    #                     direction = 1
    #                 else:
    #                     direction = -1

    #                 if ty == 0:
    #                     pointer_x = points[i][0]
    #                 else:
    #                     pointer_y = points[i][1]

    #                 for __ in range(sample_num-1):
    #                     if ty == 0:
    #                         pointer_x = pointer_x+ interval
    #                         pointer_y = k*pointer_x + b
    #                         pointer_ty = pointer_x
    #                     else:
    #                         pointer_y = pointer_y+ interval
    #                         pointer_x = b
    #                         pointer_ty = pointer_y
    #                     if (direction*points[i+1][ty])>(direction*pointer_ty+direction/2):
    #                         re_point = -1
    #                         re_index = -1
    #                         haha = -1
    #                         d_min = float('inf')
    #                         for index, pt in enumerate(contour):
    #                             if pt not in re_segment:
    #                                 d_d = distance(pointer_x,pointer_y,pt[0],pt[1])
    #                                 if d_d <3 and d_d < d_min:
    #                                     # print("verts, segment, alternative points", [points[i],points[i+1]],(pointer_x,pointer_y), pt )
    #                                     if (pt[0]!= points[i][0]) or (pt[1]!= points[i][1]):
    #                                         if (pt[0]!= points[i+1][0]) or (pt[1]!= points[i+1][1]):
    #                                             haha = 1
    #                                             if 0 <= self.segment_orientation(points[i],points[i+1],re_segment[-1],pt) < 90.5:
    #                                                 if self.poly_not_cross_open(re_segment, pt) == True:
    #                                                     if i == (len(points)-2): #这部分是临时加的，因为如果没有这个判断，会导致添加的下一个顶点无法保证与上采样的点不相交。但这样一加计算量增大挺多
    #                                                         if self.poly_not_cross_open(re_segment+[pt], points[i+1], close_flag=1) == True:
    #                                                             d_min = d_d
    #                                                             re_point = pt
    #                                                             re_index = index
    #                                                         else:
    #                                                             print("[] The point bring bad vertex 1", re_segment[0], re_segment[-1], pt, points[i+1])
    #                                                     else:
    #                                                         if self.poly_not_cross_open(re_segment+[pt], points[i+1]) == True:
    #                                                             d_min = d_d
    #                                                             re_point = pt
    #                                                             re_index = index
    #                                                         else:
    #                                                             print("[] The point bring bad vertex 2", re_segment[0], re_segment[-1], pt, points[i+1])
    #                                                 else:
    #                                                     # print("verts, segment, alternative points", [points[i],points[i+1]],(pointer_x,pointer_y), pt )
    #                                                     # print("self.poly_not_cross_open(re_segment, pt) == false")
    #                                                     pass
                                                        
    #                                             else:
    #                                                 # print("verts, segment, alternative points", [points[i],points[i+1]],(pointer_x,pointer_y), pt )
    #                                                 # print("self.segment_orientation", self.segment_orientation(points[i],points[i+1],re_segment[-1],pt))
    #                                                 pass
    #                         # print("finded point: ", re_point)

    #                         if re_index >= 0:
    #                             re_segment.append(re_point)
    #                         # else:
    #                         #     print("There is no proper point ", "in ", [points[i],points[i+1]])
    #                         #     if haha == -1:
    #                         #         print("Inappropriate distance")

    #         re_segment.append(points[i+1])
    #         re_areas.append(re_segment)
    #         print("room", point_group.index(points)," points: ", len(points), "-->", len(re_segment) )

    #     return re_areas








    def draw_graphic_iscretization(self,type_0,vertex0,direction_0,map_0):
        # print("2-1")
        re_draw = self.graphic_iscretization(type_0,vertex0,direction_0)
        if re_draw == []:
            return True
        if re_draw[0] == 1:
            for rectangle_ver in re_draw[1]:
                cv2.fillPoly(map_0, [np.array(rectangle_ver,np.int32)], self.map_color["obstacle"])
        else:
            for rectangle_ver in re_draw[2]:
                # print("hhh: ",rectangle_ver[0],rectangle_ver[1],type_0,rectangle_ver,re_draw[2])
                cv2.ellipse(map_0, (rectangle_ver[0],rectangle_ver[1]), (int(re_draw[1][0]),int(re_draw[1][1])), direction_0, 0,360, self.map_color["obstacle"],thickness=-1)
        return False
    def add_obstacle_close_wall(self):
        if self.obs_num == 0:  # No obstacle added.
            return False

        # cv2.fillPoly(editable_map, np.array(self.doorway_points,np.int32), self.map_color["obstacle"])

        
        one_map_room_num = 0


        # obs_sizeRange = [[30,50],[12,30],[5,12],[1,5]]
        obs_sizeRange = [[4,12],[12,30],[30,50]]
        disturb_obs_sizeRange = [1,5]
        # Regular wall-oriented close furniture ---obstacle_A
        sensor_type = np.random.choice([0,1],p=[0.7,0.3]) # 0:2-d laser 1:3d overview

        print("Time-begin: 0")
        self.start_time = time.time()
        self.ll_time = time.time()

        # print("$$$生成门$$$")
        doorway_obastacle_testing = []
        door_plank_testing = []
        door_number_ = 0
        for door_points_draw in self.door_points:
            
            # print("origin_doorway {}: ".format(door_number_), door_points_draw)
            door_verts_four_generation = self.Rectangle_generation(door_points_draw[0][0], door_points_draw[0][1],door_points_draw[1][0],door_points_draw[1][1], self.robotRadius*5)
            door_1st_generation = Polygon(door_verts_four_generation[0])
            door_2nd_generation = Polygon(door_verts_four_generation[1])
            if door_1st_generation.is_valid == False or door_2nd_generation.is_valid == False:
                continue
            doorway_obastacle_testing.append(door_1st_generation)
            doorway_obastacle_testing.append(door_2nd_generation)
            if np.random.choice([1,0],p=[0.8,0.2]):
                door_orientation = np.random.choice([0,1,2,3])
                if door_orientation == 0:
                    door_hinge =  door_points_draw[0]
                    door_handle_close = door_points_draw[1]
                    door_direction = -1
                elif door_orientation == 1:
                    door_hinge =  door_points_draw[0]
                    door_handle_close = door_points_draw[1]
                    door_direction = 1
                elif door_orientation == 2:
                    door_hinge =  door_points_draw[1]
                    door_handle_close = door_points_draw[0]
                    door_direction = -1
                elif door_orientation == 3:
                    door_hinge =  door_points_draw[1]
                    door_handle_close = door_points_draw[0]
                    door_direction = 1
                for __II__ in range(60):
                    thickness_door = np.random.choice([1,2,3,4],p=[0.3,0.4,0.2,0.1])
                    length_door = math.sqrt((door_points_draw[1][0]-door_points_draw[0][0])*(door_points_draw[1][0]-door_points_draw[0][0])+(door_points_draw[1][1]-door_points_draw[0][1])*(door_points_draw[1][1]-door_points_draw[0][1]))
                    
                    door_open = np.random.choice([1,0],p=[0.3,0.7])
                    if door_open == 1:
                        door_open = 0
                    else:
                        door_open = np.random.random() * math.pi/2

                    close_orientation = math.atan2(door_handle_close[1]-door_hinge[1],door_handle_close[0]-door_hinge[0])
                    if door_direction == -1:
                        open_orientation = math.pi/2 + door_open
                    else:
                        open_orientation = -math.pi/2 - door_open
                    # if open_orientation > math.pi:
                    #     open_orientation = open_orientation - 2*math.pi
                    # if open_orientation <= - math.pi:
                    #     open_orientation = open_orientation + 2*math.pi
                    rrrrr = util.transform_coord(np.array([door_handle_close[1]]),np.array([door_handle_close[0]]), np.array([door_hinge[1],door_hinge[0]]),np.array([0,0,open_orientation]))
                    door_handle_open = [int(np.round(rrrrr[1][0])), int(np.round(rrrrr[0][0]))]
                    # print("ggg: ", door_hinge)
                    # door_handle_open=[int(round(length_door * math.cos(open_orientation))), int(round(length_door * math.sin(open_orientation)))]
                    door_verts_open_eight = self.Rectangle_generation(door_hinge[0], door_hinge[1],door_handle_open[0],door_handle_open[1], thickness_door)
                    # print("eee: ", door_verts_open_eight)
                    door_angle_choose = math.atan2(door_verts_open_eight[0][2][1]-door_verts_open_eight[0][0][1],door_verts_open_eight[0][2][0]-door_verts_open_eight[0][0][0])
                    door_verts_open_four = []
                    #选取门的另外两个点
                    #-----------这一部分是判断某个射线相对于另一个射线偏转方向的示例代码
                    
                    open_orientation = open_orientation + close_orientation
                    # print(open_orientation," = ", close_orientation," + ", -door_direction * math.pi/2)
                    if open_orientation > math.pi:
                        open_orientation = open_orientation - 2*math.pi
                    if open_orientation <= - math.pi:
                        open_orientation = open_orientation + 2*math.pi
                    angle_uuu = door_angle_choose - open_orientation
                    if angle_uuu > math.pi:
                        angle_uuu = angle_uuu - 2*math.pi
                    if angle_uuu <= - math.pi:
                        angle_uuu = angle_uuu + 2*math.pi
                    if angle_uuu * door_direction >= 0:
                        door_verts_open_four = door_verts_open_eight[0]
                    else:
                        door_verts_open_four = door_verts_open_eight[1]

                    # print("door param: ", door_orientation,door_handle_close, door_hinge, door_handle_open, door_direction, open_orientation, close_orientation, )
                    #---------------
                    #判断选取的两个点是否偏向错误
                    if door_verts_open_four != []:
                        door_angle_choose2 = math.atan2(door_verts_open_four[2][1]-door_verts_open_four[0][1],door_verts_open_four[2][0]-door_verts_open_four[0][0])
                        angle_uuu2 = door_angle_choose2 - close_orientation
                        if angle_uuu2 > math.pi:
                            angle_uuu2 = angle_uuu2 - 2*math.pi
                        if angle_uuu2 <= - math.pi:
                            angle_uuu2 = angle_uuu2 + 2*math.pi
                        if angle_uuu2 * door_direction >= 0:
                            door_verts_open_four = []
                        # print("test1: ", door_verts_open_four, door_angle_choose2, close_orientation)
                    # #判断选取的两个点是否影响通行
                    if door_verts_open_four != []:
                        door_angle_choose3 = math.atan2(door_verts_open_four[2][1]-door_verts_open_four[3][1],door_verts_open_four[2][0]-door_verts_open_four[3][0])
                        angle_uuu3 = door_angle_choose3 - close_orientation
                        if angle_uuu3 > math.pi:
                            angle_uuu3 = angle_uuu3 - 2*math.pi
                        if angle_uuu3 <= - math.pi:
                            angle_uuu3 = angle_uuu3 + 2*math.pi
                        if abs(angle_uuu3) <= 89 * (math.pi / 180):
                            door_verts_open_four = []
                    else:
                        print("EEE2")
                        continue
                    #验证门是否穿越房间,这个验证可能有问题
                    if door_verts_open_four != []:
                        collide_num = 0
                        all_num = 0
                        rrrrr2 = util.transform_coord(np.array([door_handle_close[1]]),np.array([door_handle_close[0]]), np.array([door_hinge[1],door_hinge[0]]),np.array([0,0,angle_uuu2 / 2.0]))
                        door_handle_open2 = [int(np.round(rrrrr2[1][0])), int(np.round(rrrrr2[0][0]))]
                        door_area_ = Polygon([door_verts_open_four[0],door_verts_open_four[1],door_verts_open_four[2], door_handle_open2, door_handle_close, door_verts_open_four[3],door_verts_open_four[0]])
                        minx, miny, maxx, maxy = int(round(door_area_.bounds[0])), int(round(door_area_.bounds[1])), int(round(door_area_.bounds[2])), int(round(door_area_.bounds[3]))
                        # print("minx, miny, maxx, maxy", minx, miny, maxx, maxy)
                        # print("shape self.world ", self.world.shape)
                        # print("length: ",length_door)
                        for i in range(minx, maxx + 1):
                            for j in range(miny, maxy + 1):
                                Point_d = Point(i,j)
                                if Point_d.within(door_area_) == True:
                                    if (i != door_hinge[0] or j != door_hinge[1]) and (i != door_handle_close[0] or j != door_handle_close[1]):
                                        if (j >= self.world.shape[0] or i >= self.world.shape[1]) or (j < 0 or i <0):
                                            collide_num = thickness_door + 5
                                            break
                                        if self.world[j][i] == self.map_color['obstacle']:
                                            collide_num = collide_num + 1
                        if collide_num > thickness_door:
                            door_verts_open_four = []
                    else:
                        print("EEE3")
                        continue
                    if door_verts_open_four != []:
                        door_shape_for_testing = Polygon([door_verts_open_four[0],door_verts_open_four[1],door_verts_open_four[2], door_handle_open2, door_handle_close, door_verts_open_four[3],door_verts_open_four[0]])
                        if door_shape_for_testing.is_valid == False:
                            continue
                        else:
                            cv2.fillPoly(self.world, [np.array(door_verts_open_four,np.int32)], self.map_color["obstacle"])
                            # door_plank_testing.append(Polygon([door_verts_open_four[0],door_verts_open_four[1],door_verts_open_four[2], door_verts_open_four[3],door_verts_open_four[0]]))
                            door_plank_testing.append(door_shape_for_testing)
                            # print("door {}: ".format(door_number_), door_verts_open_four)
                            # print("length: ",length_door, math.sqrt((door_handle_open[0]-door_hinge[0])*(door_handle_open[0]-door_hinge[0])+(door_handle_open[1]-door_hinge[1])*(door_handle_open[1]-door_hinge[1])))
                            break
                    else:
                        # print("EEE4")
                        pass
            door_number_ = door_number_ + 1

        end_time = time.time()
        elapsed_time = end_time - self.start_time
        elapsed_time2 = end_time - self.ll_time
        print("Time-door: {:.2f}s-{:.2f}s".format(elapsed_time, elapsed_time2))
        self.ll_time = time.time()
        noise_info_ = []
        

        All_B_contours = []

        all_point_lasers_home = []
        for index_test, points_test_segmentation in enumerate(self.segmentation_points):
            main_directions = []
            B_contours = []
            # print("index_test: ",index_test)
            one_map_room_num = one_map_room_num + 1
            print("\n###  第{}个房间  ###".format(one_map_room_num))
            # print("Time-building: 0")
            building_start_time = time.time()
            building_ll_time = time.time()
            # input("等待继续...")
            self.free_space_obstacles = Polygon(points_test_segmentation)
            room_annotion_is_valid = self.free_space_obstacles.is_valid
            if room_annotion_is_valid == False:
                print("\033[43;34m房间标注的边界无效\033[0m")
                return False
            added_obstacles_list = []
            added_obstacles_list_B = []
            boundary_free_list = [list(range(len(points_test_segmentation)))]
            area = self.segmentation_area_properties[index_test]["new_area"]
            area = float(area) / self.config['meter2pixel'] / self.config['meter2pixel']
            # possible area range :3-80
            num_range = [[int(round(area/15)),int(round(area/4))], [int(round(area/30)),int(round(area/2))], [0,int(round(area/1))]]
            # num_range = [[int(round(area/20)),int(round(area/12))], [int(round(area/12)),int(round(area/8))], [0,int(round(area/4))]]
            

            boundary_append_obstacle_points = {}
            boundary_append_obstacle_points_count = -1
            boundary_append_obstacle_points_list = list(range(len(points_test_segmentation)))


            # print("**参数(1):全局障碍物数量**")
            # print("房间面积: ",area)
            # print("-面积引导的障碍物数量范围基准：", num_range)

            num_range_result = np.random.choice([0,1,2],p=[0.2,0.6,0.2])
            # num_range_result = np.random.choice([0,1,2],p=[0.35,0.45,0.2])
            # seed_num = [0,1,3]

            # print("-障碍物数量范围随机种子：", num_range_result,num_range[num_range_result])
            # print("(范围基准：", num_range[0],")")
            #area =3:num_range[(0),(0~1),(0~3)];;;[(0),(1),(2~3)]
            #area =4:num_range[(0~1),(0~2),(0~4)];;;[(0~1),(2),(3~4)]
            #area =5:num_range[(0~1),(0~2),(0~5)];;;[(0~1),(2),(3~5)]
            #area =6:num_range[(0~1),(0~3),(0~6)];;;[(0~1),(2~3),(4~6)]
            #area =40:num_range[(3~10),(1~20),(0~40)];;;[(3~10),(1~2,11~20),(0,21~40)]
            #area =80:num_range[(5~20),(3~40),(0~80)];;;[(5~20),(3~4,21~40),(0~2,41~80)]
            # print("num_range_result: ",num_range_result)
            if num_range_result == 0:
                if num_range[0][0] == num_range[0][1]:
                    obstacle_num = num_range[0][0]
                else:
                    obstacle_num = int(np.random.randint(num_range[0][0],num_range[0][1]+1))
            else:
                if num_range[num_range_result][0] == num_range[num_range_result][1]:
                    if (num_range[num_range_result][0] <= num_range[num_range_result-1][0]) and (num_range[num_range_result][1] >= num_range[num_range_result-1][1]):
                        tmp=[]
                    else:
                        tmp = [num_range[num_range_result][0]]
                else:
                    if num_range[num_range_result][0] < num_range[num_range_result-1][0]:
                        tmp1 = list(range(num_range[num_range_result][0], num_range[num_range_result-1][0]))
                    else:
                        tmp1 = []
                    if num_range[num_range_result][1] > num_range[num_range_result-1][1]:
                        tmp2 = list(range(num_range[num_range_result-1][1]+1, num_range[num_range_result][1]+1))
                    else:
                        tmp2 = []
                    tmp = tmp1 + tmp2
                if tmp == []:
                    obstacle_num = int(np.random.randint(num_range[0][0],num_range[0][1]+1))
                else:
                    tmp_index = len(tmp)
                    obstacle_num_index = int(np.random.randint(0,tmp_index))
                    obstacle_num = tmp[obstacle_num_index]

            # print("---全局确定的障碍物数量：", obstacle_num,"\n")
            # print("**参数(2):全局障碍物尺寸范围")
            # print("-障碍物尺寸基准:",obs_sizeRange)
            obs_sizeRange_room = []
            # area_limit = max((area*0.8), area-1.5)
            # # Effect of Area on Obstacle Size
            # length_limit = area_limit/obs_sizeRange[2][0]
            bbox = self.free_space_obstacles.bounds
            min_x,min_y,max_x,max_y = bbox[0],bbox[1],bbox[2],bbox[3]
            length_limit = math.sqrt((max_x-min_x)*(max_x-min_x)+(max_y-min_y)*(max_y-min_y))
            # length_limit = max((max_x-min_x), (max_y-min_y))
            # print("-房间长宽：",max((max_x-min_x), (max_y-min_y)), min((max_x-min_x), (max_y-min_y)),"房间尺寸对障碍物尺寸限制：",length_limit)
            # 目前这个房间尺寸限制是用的对角线，限制还是很宽松的
            for seg_range in obs_sizeRange:
                if length_limit < seg_range[1]:
                    if length_limit <= seg_range[0]:
                        pass
                    else:
                        obs_sizeRange_room.append([seg_range[0], area_limit])
                else:
                    obs_sizeRange_room.append(seg_range)
            if len(obs_sizeRange_room) == 0:
                break
            # elif len(obs_sizeRange_room) == 1:
            #     break
            
            
            
            # print("**参数(3):障碍物类型-数量分配")
            # A是贴墙类障碍物/装修，B是独立障碍物/装修，设置两类障碍物数量
            if obstacle_num > 3: 
                obstacle_A_num_boundary_midle = int(round(obstacle_num/float(2)))
                if obstacle_A_num_boundary_midle > obstacle_num:
                    obstacle_A_num_boundary_midle = obstacle_num
                if np.random.choice([0,1],p=[0.7,0.3]) == 0:
                    obstacle_A_num = np.random.randint(obstacle_A_num_boundary_midle,obstacle_num+1)
                else:
                    obstacle_A_num = np.random.randint(0,obstacle_A_num_boundary_midle)
            else:
                if obstacle_num == 0:
                    obstacle_A_num = 0 #暂时允许没有障碍物
                else:
                    obstacle_A_num = np.random.randint(0,obstacle_num+1)
            obstacle_B_num = obstacle_num - obstacle_A_num
            # obstacle_C_num = 5
            # obstacle_B_num = 2  #test
            # obstacle_A_num = 5  #test


            fake_flag_obstacle_A = False
            if obstacle_A_num == 0:
                fake_flag_obstacle_A = True
                obstacle_A_num = 1
                obstacle_C_num = 0
            else:
                obstacle_C_num = np.random.randint(0,obstacle_A_num+1)
            

    
            # obstacle_C_num = 2
            # print("---全局障碍物尺寸范围:",obs_sizeRange_room)
            # print("---A类型障碍物(靠墙)预配置数量:",obstacle_A_num)
            # print("---B类型障碍物(独立)预配置数量:",obstacle_B_num)
            # print("---C类型障碍物(靠墙但倾斜)预配置数量:",obstacle_C_num)
            print("障碍物总数:",obstacle_num," A: ", obstacle_A_num, "B: ", obstacle_B_num, "C: ", obstacle_C_num)
            # input("等待继续...")


            end_time2 = time.time()
            elapsed_time3 = end_time2 - building_start_time
            elapsed_time4 = end_time2 - building_ll_time
            print("Time-IN-param: {:.2f}-{:.2f}".format(elapsed_time4, elapsed_time3))
            building_ll_time = time.time()
                
            



            print("---A类型障碍物生成", obstacle_A_num)
            # print("---C类型障碍物(靠墙但倾斜)预配置数量:",obstacle_C_num)
            # 设置A类障碍物类型，1矩形贴墙障碍物 2奇异形状沿墙贴墙障碍物，3不沿房间贴墙障碍物
            point_num_obstacles = 0
            for _ in range(obstacle_A_num):
                # input("等待继续...")

                searching_times_1 = 0
                point_num_obstacles = point_num_obstacles + 1
                # print("\n$$ {}/{} $$".format(point_num_obstacles,obstacle_A_num))
                if obstacle_C_num == 0:
                    obs_C_flag = False
                else:
                    if (obstacle_A_num - point_num_obstacles) < obstacle_C_num:
                        obs_C_flag = True
                    else:
                        obs_C_flag = False
                # input("等待继续...")
                # print("开始生成贴边障碍物".format(point_num_obstacles,obstacle_A_num))
                # area_limit_1 = self.segmentation_area_properties[index_test]["new_area"]*2/3
                # area_limit_2 = self.segmentation_area_properties[index_test]["new_area"] - 2.5*self.config['meter2pixel']*self.config['meter2pixel']
                # area_limit = min(area_limit_1, area_limit_2)
                re_find_A = False
                obs_type_A = np.random.choice([0,1,2],p=[0.6,0.2,0.2])
                # obs_type_A=0 #test

                A_other_direction = np.random.choice([0,1],p=[0.9,0.1])

                if sensor_type == 0:
                    flag_2d_laser_lattice = np.random.choice([0,1],p=[0.5,0.5])
                else:
                    flag_2d_laser_lattice = 0




                is_obstacles_close = np.random.choice([0,1],p=[0.8,0.2])
                # if is_obstacles_close == 0:
                #     print("此障碍物不接触其他障碍物")
                # else:
                #     print("此障碍物可能接触其他障碍物")


                # print("---整体形状(0矩形;1椭圆形;2三角形):",obs_type_A)
                while searching_times_1 < 15:
                    # print("      {}/15".format(searching_times_1))
                    obs_type_A_a_size, obs_type_A_b_size= self.random_multi_obstacle_size(0, obs_type_A, obs_sizeRange_room)
                    


                    area = self.segmentation_area_properties[index_test]["new_area"]
                    area_limit = min((area*0.7), area-(1.5*self.config['meter2pixel']*self.config['meter2pixel']))
                    obs_type_A_area = obs_type_A_a_size * obs_type_A_b_size
                    cout_area_limit_searching = 1
                    go_next_flag = True
                    
                    while (obs_type_A_area > area_limit) and cout_area_limit_searching < 251:
                        # print("**---重新确定尺寸**")
                        obs_type_A_a_size, obs_type_A_b_size= self.random_multi_obstacle_size(0, obs_type_A, obs_sizeRange_room)
                        obs_type_A_area = obs_type_A_a_size * obs_type_A_b_size
                        # print("area_limit,obs_type_A_area: ", area_limit,obs_type_A_area)
                        if cout_area_limit_searching % 50 == 0:
                            obs_sizeRange_room_ttt = []
                            length_limit = length_limit * 0.8
                            for seg_range in obs_sizeRange_room:
                                if length_limit < seg_range[1]:
                                    if length_limit <= seg_range[0]:
                                        pass
                                    else:
                                        obs_sizeRange_room_ttt.append([seg_range[0], area_limit])
                                else:
                                    obs_sizeRange_room_ttt.append(seg_range)
                            obs_sizeRange_room = obs_sizeRange_room_ttt
                        cout_area_limit_searching = cout_area_limit_searching + 1
                        if len(obs_sizeRange_room) == 0:
                            go_next_flag = False
                    

                    if cout_area_limit_searching == 251:
                        go_next_flag = False
                    if go_next_flag == False:
                        break 

                    if (obs_C_flag == True) and (obs_type_A == 0):
                        obs_y_c = 0
                        obs_x_c = 0
                        obstacle_C_theta = np.random.random()*360
                        cthe_c = math.cos(np.pi/180* obstacle_C_theta)
                        sthe_c = math.sin(np.pi/180* obstacle_C_theta)
                        obs_type_A_a_size = obs_type_A_a_size/2
                        obs_type_A_b_size = obs_type_A_b_size/2
                        # rect_b = [[obs_x_b + (-obs_type_B_a_size * cthe_b - -obs_type_B_b_size * sthe_b), obs_y_b + (-obs_type_B_a_size * sthe_b + -obs_type_B_b_size * cthe_b)],
                        #                 [obs_x_b + (-obs_type_B_a_size * cthe_b - obs_type_B_b_size * sthe_b), obs_y_b + (-obs_type_B_a_size * sthe_b + obs_type_B_b_size * cthe_b)],
                        #                 [obs_x_b + (obs_type_B_a_size * cthe_b - obs_type_B_b_size * sthe_b), obs_y_b + (obs_type_B_a_size * sthe_b + obs_type_B_b_size * cthe_b)],
                        #                 [obs_x_b + (obs_type_B_a_size * cthe_b - -obs_type_B_b_size * sthe_b), obs_y_b + (obs_type_B_a_size * sthe_b + -obs_type_B_b_size * cthe_b)]]
                        rect_c = [[int(round(obs_x_c + (-obs_type_A_a_size * cthe_c - -obs_type_A_b_size * sthe_c))), int(round(obs_y_c + (-obs_type_A_a_size * sthe_c + -obs_type_A_b_size * cthe_c)))],
                                        [int(round(obs_x_c + (-obs_type_A_a_size * cthe_c - obs_type_A_b_size * sthe_c))), int(round(obs_y_c + (-obs_type_A_a_size * sthe_c + obs_type_A_b_size * cthe_c)))],
                                        [int(round(obs_x_c + (obs_type_A_a_size * cthe_c - obs_type_A_b_size * sthe_c))), int(round(obs_y_c + (obs_type_A_a_size * sthe_c + obs_type_A_b_size * cthe_c)))],
                                        [int(round(obs_x_c + (obs_type_A_a_size * cthe_c - -obs_type_A_b_size * sthe_c))), int(round(obs_y_c + (obs_type_A_a_size * sthe_c + -obs_type_A_b_size * cthe_c)))]]
                        
                        min_x_rect_c = min(rect_c[0][0],rect_c[1][0],rect_c[2][0],rect_c[3][0])
                        min_y_rect_c = min(rect_c[0][1],rect_c[1][1],rect_c[2][1],rect_c[3][1])
                        max_x_rect_c = max(rect_c[0][0],rect_c[1][0],rect_c[2][0],rect_c[3][0])
                        max_y_rect_c = max(rect_c[0][1],rect_c[1][1],rect_c[2][1],rect_c[3][1])
                        rect_c_pos = []

                        rect_c_copy = copy.deepcopy(rect_c)
                        
                        # flag_search_uuu = False
                        temp_rect_c_pos_el = []
                        for index_uuu,i_uuu in enumerate(rect_c_copy):
                            if i_uuu[0] == min_x_rect_c:
                                temp_rect_c_pos_el.append(i_uuu[1]-min_y_rect_c)
                                rect_c_copy.pop(index_uuu)
                                # flag_search_uuu = True
                        if len(temp_rect_c_pos_el) > 1:
                            temp_rect_c_pos_el.sort()
                        rect_c_pos.append(temp_rect_c_pos_el)
                        temp_rect_c_pos_el = []
                        for index_uuu,i_uuu in enumerate(rect_c_copy):
                            if i_uuu[1] == max_y_rect_c:
                                temp_rect_c_pos_el.append(i_uuu[0]-min_x_rect_c)
                                rect_c_copy.pop(index_uuu)
                        if len(temp_rect_c_pos_el) > 1:
                            temp_rect_c_pos_el.sort()
                        rect_c_pos.append(temp_rect_c_pos_el)
                        temp_rect_c_pos_el = []
                        for index_uuu,i_uuu in enumerate(rect_c_copy):
                            if i_uuu[0] == max_x_rect_c:
                                temp_rect_c_pos_el.append(max_y_rect_c - i_uuu[1])
                                rect_c_copy.pop(index_uuu)
                        if len(temp_rect_c_pos_el) > 1:
                            temp_rect_c_pos_el.sort()
                        rect_c_pos.append(temp_rect_c_pos_el)
                        temp_rect_c_pos_el = []
                        for index_uuu,i_uuu in enumerate(rect_c_copy):
                            if i_uuu[1] == min_y_rect_c:
                                temp_rect_c_pos_el.append(max_x_rect_c - i_uuu[0])
                                rect_c_copy.pop(index_uuu)
                        if len(temp_rect_c_pos_el) > 1:
                            temp_rect_c_pos_el.sort()
                        rect_c_pos.append(temp_rect_c_pos_el)
                        temp_rect_c_pos_el = []

                        if len(rect_c_copy) == 0 and len(rect_c_pos) == 4:
                            if (max_x_rect_c - min_x_rect_c) >= (max_y_rect_c - min_y_rect_c):
                                obs_type_A_a_size = (max_x_rect_c - min_x_rect_c)
                                obs_type_A_b_size = (max_y_rect_c - min_y_rect_c)
                                rect_c_pos.append(1)
                            else:
                                obs_type_A_a_size = (max_y_rect_c - min_y_rect_c)
                                obs_type_A_b_size = (max_x_rect_c - min_x_rect_c)
                                rect_c_pos.append(0)
                        else:
                            rect_c_pos = []
                    if (obs_C_flag == True) and (obs_type_A == 1):
                        obs_y_c = 0
                        obs_x_c = 0
                        obstacle_C_theta = np.random.random()*360
                        dia_c = max(obs_type_A_a_size,obs_type_A_b_size) + 20
                        obs_type_A_a_size = obs_type_A_a_size/2
                        obs_type_A_b_size = obs_type_A_b_size/2
                        
                        im_c = np.zeros([dia_c, dia_c], dtype=np.uint8)
                        center_c_1 = int(round(dia_c/2.0))
                        cv2.ellipse(im_c, (center_c_1,center_c_1), (int(round(obs_type_A_a_size)),int(round(obs_type_A_b_size))), obstacle_C_theta, 0,360, self.map_color["obstacle"],thickness=-1)
                        x_boundingRect, y_boundingRect, w_boundingRect, h_boundingRect = cv2.boundingRect(im_c)
                        ellipse_c_pos = []
                        if (w_boundingRect > 0) and (h_boundingRect > 0) and (x_boundingRect <= center_c_1) and (y_boundingRect <= center_c_1):
                            if w_boundingRect >= h_boundingRect:
                                ellipse_c_pos.append(center_c_1 - x_boundingRect)
                                ellipse_c_pos.append(center_c_1 - y_boundingRect)
                                ellipse_c_pos.append(1)
                                ellipse_c_pos.append([copy.deepcopy(obs_type_A_a_size), copy.deepcopy(obs_type_A_b_size)])
                                ellipse_c_pos.append(obstacle_C_theta)
                                obs_type_A_a_size = w_boundingRect
                                obs_type_A_b_size = h_boundingRect
                            else:
                                ellipse_c_pos.append(center_c_1 - x_boundingRect)
                                ellipse_c_pos.append(center_c_1 - y_boundingRect)
                                ellipse_c_pos.append(0)
                                ellipse_c_pos.append([copy.deepcopy(obs_type_A_a_size), copy.deepcopy(obs_type_A_b_size)])
                                ellipse_c_pos.append(obstacle_C_theta)
                                obs_type_A_a_size = h_boundingRect
                                obs_type_A_b_size = w_boundingRect
                        # print("C-ellipse-test-ellipse_c_pos", ellipse_c_pos)

                    searching_times_2 = 0
                    type_set = []
                    
                    while searching_times_2 < 3:
                        close_side_type = np.random.choice([0,1,2],p=[0.4,0.4,0.2])
                        if len(type_set)==3:
                            searching_times_2 = 3
                            break
                        while close_side_type in type_set:
                            close_side_type = np.random.choice([0,1,2],p=[0.4,0.4,0.2])
                        
                        if close_side_type == 1:
                            close_side_size = obs_type_A_b_size
                            vertical_side_size = obs_type_A_a_size
                        if close_side_type == 0:
                            close_side_size = obs_type_A_a_size
                            vertical_side_size = obs_type_A_b_size
                        if close_side_type == 2:
                            close_side_size = obs_type_A_a_size # test
                            vertical_side_size = obs_type_A_b_size
                        


                    
                        find_ok = False
                        for i_segta in range(len(points_test_segmentation)*2):
                            len_boundary_free_list = 0
                            for v in boundary_free_list:
                                len_boundary_free_list = len_boundary_free_list + len(v)
                            index_for_corner = np.random.randint(0,len_boundary_free_list)
                            kk2 = index_for_corner
                            kk1 = 0
                            # print("index_for_corner: ",index_for_corner)

                            flag_kk1_kk2 = False
                            for index_v, v in enumerate(boundary_free_list):
                                if kk2 >= len(v):
                                    kk2 = kk2 - len(v)
                                else:
                                    flag_kk1_kk2 = True
                                    kk1 = index_v
                                    break
                            
                            if flag_kk1_kk2 == False:
                                print("随机顶点出错")
                                break
                            # print("kk1,kk2: ",kk1,kk2)
                            # print(len(boundary_free_list),len(boundary_free_list[kk1]))
                            index_for_corner = boundary_free_list[kk1][kk2]

                            index_for_corner_range = len(boundary_free_list[kk1])     
                                  
                            
                            begin_index_copy = copy.deepcopy(index_for_corner)
                            close_side_corner_begin = points_test_segmentation[index_for_corner]
                            begin_kk1 = copy.deepcopy(kk1)
                            begin_kk2 = copy.deepcopy(kk2)
                            
                            
                            
                            possible_corner = []
                            max_boundary_coincidence = 0
                            min_dis_end_corner = 0

                            
                            while kk2 < (index_for_corner_range-1):
                                kk2 = kk2 + 1
                                index_for_corner = boundary_free_list[kk1][kk2]
                            # while index_for_corner <len(points_test_segmentation):
                            #     index_for_corner = index_for_corner + 1
                                dis_end_corner_tmp = math.sqrt((points_test_segmentation[index_for_corner][0]-close_side_corner_begin[0])*(points_test_segmentation[index_for_corner][0]-close_side_corner_begin[0])+(points_test_segmentation[index_for_corner][1]-close_side_corner_begin[1])*(points_test_segmentation[index_for_corner][1]-close_side_corner_begin[1]))
                                # print("距离增值......")
                                if (dis_end_corner_tmp >= close_side_size) and (abs(dis_end_corner_tmp-close_side_size) < 4):
                                    
                                    #这个地方的判断条件是：构建的边的长度大于等于要求长度且和要求长度差别在4个像素内。这里这个条件后期可以改
                                    cout_all__interval_retri = 0
                                    cout_ok__interval_retri = 0
                                    
                                    # print("test_error: ", begin_index_copy+1,index_for_corner)
                                    for Boundary_interval_retri in range(begin_index_copy+1,index_for_corner):
                                        if self.is_in_line(close_side_corner_begin[0],close_side_corner_begin[1],points_test_segmentation[index_for_corner][0],points_test_segmentation[index_for_corner][1],points_test_segmentation[Boundary_interval_retri][0],points_test_segmentation[Boundary_interval_retri][1]):
                                            cout_ok__interval_retri = cout_ok__interval_retri + 1
                                        cout_all__interval_retri = cout_all__interval_retri + 1
                                    if cout_all__interval_retri == 0:
                                        boundary_coincidence_rate = 0
                                    else:
                                        boundary_coincidence_rate = float(cout_ok__interval_retri) / cout_all__interval_retri
                                    # print("候选结束点: ",index_for_corner, dis_end_corner_tmp, boundary_coincidence_rate,points_test_segmentation[index_for_corner])
                                    if boundary_coincidence_rate > 0.9:
                                        if (boundary_coincidence_rate > max_boundary_coincidence) or ((boundary_coincidence_rate == max_boundary_coincidence) and (dis_end_corner_tmp < min_dis_end_corner)):
                                            possible_corner = [close_side_corner_begin,points_test_segmentation[index_for_corner]]
                                            max_boundary_coincidence = copy.deepcopy(boundary_coincidence_rate)
                                            min_dis_end_corner = copy.deepcopy(dis_end_corner_tmp)
                                            end_kk1 = copy.deepcopy(kk1)
                                            end_kk2 = copy.deepcopy(kk2)
                                    ##如果begin到end正好转了一圈，就会把没用上的点给包括进去，类似于劣弧。这时候依据boundary_coincidence_rate比例就可以排除劣弧。也就是此处只允许优弧。

                            
                                    #(1)保证图形全在房间(2)保证房间与墙的重合面积足够大(3)保证图形与障碍物没有交点(4)保证空余面积足够
                                    #问题一如何从已构建的障碍物的地图中选择边界，如何处理原边界与障碍物引起的边界
                                    #问题而如何计算与已有障碍物的交点以及空余面积

                            # if possible_corner == []:
                            #     print("未找到任何合适顶点")
                            
                            if possible_corner != []:
                                # 判断条件一：选择障碍物朝向
                                re_re = self.Rectangle_generation(possible_corner[0][0], possible_corner[0][1],possible_corner[1][0],possible_corner[1][1],vertical_side_size)
                                area_re1 = Polygon(re_re[0])
                                area_is_valid = area_re1.is_valid
                                valid_flag_area_12 = True
                                if area_is_valid == False:
                                    print("\033[43;34m area_re1的边界无效\033[0m")
                                    valid_flag_area_12 = False
                                else:
                                # try:
                                    intersection_t1 = area_re1.intersection(self.free_space_obstacles).area
                                    area_re2 = Polygon(re_re[1])
                                    area_is_valid_2 = area_re2.is_valid
                                    if area_is_valid_2 == False:
                                        print("\033[43;34m area_re2的边界无效\033[0m")
                                        valid_flag_area_12 = False
                                    else:
                                        intersection_t2 = area_re2.intersection(self.free_space_obstacles).area
                                # except:
                                #     # print("free_space shape tesing: ",self.free_space_obstacles.is_valid)
                                #     # print("area1 shape tesing: ",area_re1.is_valid)
                                #     # print("area2 shape tesing: ", area_re2.is_valid)
                                #     return False
                                # try:
                                #     intersection_t1 = area_re1.intersection(self.free_space_obstacles).area
                                #     area_re2 = Polygon(re_re[1])
                                #     intersection_t2 = area_re2.intersection(self.free_space_obstacles).area
                                # except:
                                #     return False
                                if valid_flag_area_12 == True:
                                    if (intersection_t1 >= intersection_t2) and (intersection_t1 > 0.95*area_re1.area):
                                        obstacle_possbile_shape_re = re_re[0]
                                        area_re_k = area_re1
                                        dire_area_re = True
                                    elif (intersection_t2 >= intersection_t1) and (intersection_t2 > 0.95*area_re2.area):
                                        obstacle_possbile_shape_re = re_re[1]
                                        area_re_k = area_re2
                                        dire_area_re = False
                                    else:
                                        obstacle_possbile_shape_re = []
                                else:
                                    obstacle_possbile_shape_re = []
                                    # print("选取顶点失败", " intersection_t1, area_re1: ", intersection_t1, area_re1.area)
                                    # print("fail 111")

                                # print("wrong_room  test000: ", Polygon(obstacle_possbile_shape_re).intersection(self.free_space_obstacles).area / Polygon(obstacle_possbile_shape_re).area)
                                 

                                
                                if obstacle_possbile_shape_re != []:
                                    flag_sufficient_distance = True
                                    if (obs_C_flag == True) and (obs_type_A == 0):
                                        new_rect_c = []
                                        if rect_c_pos != []:
                                            if (rect_c_pos[4] == 1 and ((close_side_type == 0) or (close_side_type == 2))) or (rect_c_pos[4] == 0 and close_side_type == 1):
                                                index_obstacle_possbile = 1
                                                for _rect_c_pos_ in rect_c_pos[:4]:
                                                    for c_rect_point in _rect_c_pos_:
                                                        if index_obstacle_possbile == 3:
                                                            x_new_rect_c, y_new_rect_c = self.line_segmentation_length(obstacle_possbile_shape_re[index_obstacle_possbile][0],obstacle_possbile_shape_re[index_obstacle_possbile][1],obstacle_possbile_shape_re[0][0],obstacle_possbile_shape_re[0][1],c_rect_point)
                                                        else:
                                                            x_new_rect_c, y_new_rect_c = self.line_segmentation_length(obstacle_possbile_shape_re[index_obstacle_possbile][0],obstacle_possbile_shape_re[index_obstacle_possbile][1],obstacle_possbile_shape_re[index_obstacle_possbile + 1][0],obstacle_possbile_shape_re[index_obstacle_possbile + 1][1],c_rect_point)
                                                        new_rect_c.append((x_new_rect_c, y_new_rect_c))
                                                    index_obstacle_possbile = index_obstacle_possbile + 1
                                                    if index_obstacle_possbile == 4:
                                                        index_obstacle_possbile = 0
                                            else:
                                                index_obstacle_possbile = 0
                                                for _rect_c_pos_ in rect_c_pos[:4]:
                                                    for c_rect_point in _rect_c_pos_:
                                                        if index_obstacle_possbile == 3:
                                                            x_new_rect_c, y_new_rect_c = self.line_segmentation_length(obstacle_possbile_shape_re[index_obstacle_possbile][0],obstacle_possbile_shape_re[index_obstacle_possbile][1],obstacle_possbile_shape_re[0][0],obstacle_possbile_shape_re[0][1],c_rect_point)
                                                        else:
                                                            x_new_rect_c, y_new_rect_c = self.line_segmentation_length(obstacle_possbile_shape_re[index_obstacle_possbile][0],obstacle_possbile_shape_re[index_obstacle_possbile][1],obstacle_possbile_shape_re[index_obstacle_possbile + 1][0],obstacle_possbile_shape_re[index_obstacle_possbile + 1][1],c_rect_point)
                                                        new_rect_c.append((x_new_rect_c, y_new_rect_c))
                                                    index_obstacle_possbile = index_obstacle_possbile + 1
                                                    if index_obstacle_possbile == 4:
                                                        index_obstacle_possbile = 0
                                        if len(new_rect_c) != 4:
                                            new_rect_c = []
                                        if new_rect_c == []:
                                            flag_sufficient_distance = False


                                    # 有个事得注意，下面用来判断的所有obstacle_possbile_shape_re都是粗矩形包围框，不是最终的形状
                                    # 判断条件：是否挡住门口
                                    # print("=== testing 1===") #2_11 test
                                    for doorway_polo in doorway_obastacle_testing:
                                        if doorway_polo.intersection(self.free_space_obstacles).area  > 1:
                                            if area_re_k.intersection(doorway_polo).area > 1:
                                                flag_sufficient_distance = False
                                    for doorway_polo in door_plank_testing:
                                        # print("doora: ", list(doorway_polo.exterior.coords) )
                                        if doorway_polo.intersection(self.free_space_obstacles).area  > 1:
                                            if area_re_k.intersection(doorway_polo).area >= 1:
                                                flag_sufficient_distance = False


                                    # 判断条件二：避免与该房间已有障碍物距离过近    
                                    if is_obstacles_close == 0 and obs_C_flag == False:
                                        expansive_area_re_k = area_re_k.buffer(12,join_style=2,cap_style=2)
                                    else:
                                        # expansive_area_re_k = area_re_k.buffer(1,join_style=2,cap_style=2)
                                        expansive_area_re_k = area_re_k
                                    for try_obstacle in added_obstacles_list:
                                        area_try_obstacle = Polygon(try_obstacle)
                                        if expansive_area_re_k.intersection(area_try_obstacle).area > 0:
                                            flag_sufficient_distance = False
                                            # print("已有障碍物距离过近")
                                            # print("fail 222")
                                            # break


                                    for i_list_1 in obstacle_possbile_shape_re:
                                        if i_list_1[0] < 0 or  i_list_1[1] < 0:
                                            flag_sufficient_distance = False
                                            # print("超出范围")
                                            # break
                                    # 判断条件：防止障碍物之间过近的要点之二
                                    if is_obstacles_close == 0:
                                        if len(boundary_free_list) > 1:
                                            if (begin_kk2 < 5) or (len(boundary_free_list[end_kk1])-1-end_kk2  < 5):
                                                flag_sufficient_distance = False
                                    # print("=== testing 2===") #2_11 test
                                    # 判断条件三：避免与墙距离太近堵住路,这个再考虑考虑，如果一直离着墙边远，如何模拟两面、三面靠墙的情况
                                    if flag_sufficient_distance == True:
                                        contour_cross_count = 0
                                        obstacle_range_area = Polygon(obstacle_possbile_shape_re)
                                        for i_seg_points in range(len(points_test_segmentation)):
                                            if i_seg_points not in range(begin_index_copy,index_for_corner+1):
                                                p_lot = Point(points_test_segmentation[i_seg_points][0],points_test_segmentation[i_seg_points][1])
                                                if(p_lot.within(obstacle_range_area)==True):
                                                    contour_cross_count = contour_cross_count + 1
                                                if contour_cross_count > 3:
                                                    # print("障碍物存在与边界的交点1")
                                                    flag_sufficient_distance = False
                                                    break
                                        if contour_cross_count > 3:
                                            # print("障碍物存在与边界的交点1")
                                            flag_sufficient_distance = False
                                                    # break
                                        if flag_sufficient_distance == True:
                                            re_re_expansive = self.Rectangle_generation(possible_corner[0][0], possible_corner[0][1],possible_corner[1][0],possible_corner[1][1],(vertical_side_size+8))
                                            if dire_area_re == True:
                                                re_re_expansive_2 = re_re_expansive[0]
                                                area_ori = area_re1.area
                                                intersection_ori = intersection_t1
                                            else:
                                                re_re_expansive_2 = re_re_expansive[1]
                                                area_ori = area_re2.area
                                                intersection_ori = intersection_t2
                                            re_re_expansive_2_poly = Polygon(re_re_expansive_2)
                                            # print("re_re_expansive_2: ",re_re_expansive_2)
                                            if abs((re_re_expansive_2_poly.area - re_re_expansive_2_poly.intersection(self.free_space_obstacles).area) - (area_ori - intersection_ori)) > 1:
                                                flag_sufficient_distance = False
                                                # print("障碍物存在与边界的交点2")
                                                # break

                                            if flag_sufficient_distance == True:
                                                re_re_cropping = self.Rectangle_generation(possible_corner[0][0], possible_corner[0][1],possible_corner[1][0],possible_corner[1][1],(0.2*vertical_side_size))
                                                if dire_area_re == True:
                                                    # print("re_re_cropping[0]: ",re_re_cropping[0])
                                                    area_ori_points = (re_re_cropping[0][2],re_re_cropping[0][3],re_re[0][3],re_re[0][2])
                                                    area_ori_poly = Polygon(area_ori_points)
                                                    area_ori = area_ori_poly.area
                                                    intersection_ori = area_ori_poly.intersection(self.free_space_obstacles).area
                                                    re_re_cropping_expansive_v1 = self.Rectangle_generation(re_re_cropping[0][3][0], re_re_cropping[0][3][1],re_re[0][3][0],re_re[0][3][1],8)
                                                    if (((re_re_cropping[0][3][0]-re_re_cropping[0][2][0])*(re_re_cropping[0][3][0]-re_re_cropping_expansive_v1[0][3][0]))<=0) and (((re_re_cropping[0][3][1]-re_re_cropping[0][2][1])*(re_re_cropping[0][3][1]-re_re_cropping_expansive_v1[0][3][1]))<=0):
                                                        re_re_cropping_expansive_re = [re_re_cropping_expansive_v1[0][2],re_re_cropping_expansive_v1[0][3]]
                                                    else:
                                                        re_re_cropping_expansive_re = [re_re_cropping_expansive_v1[1][2],re_re_cropping_expansive_v1[1][3]]
                                                    
                                                    re_re_cropping_expansive_v2 = self.Rectangle_generation(re_re_cropping[0][2][0], re_re_cropping[0][2][1],re_re[0][2][0],re_re[0][2][1],8)
                                                    if (((re_re_cropping[0][2][0]-re_re_cropping[0][3][0])*(re_re_cropping[0][2][0]-re_re_cropping_expansive_v2[0][3][0]))<=0) and (((re_re_cropping[0][2][1]-re_re_cropping[0][3][1])*(re_re_cropping[0][2][1]-re_re_cropping_expansive_v2[0][3][1]))<=0):
                                                        re_re_cropping_expansive_re.append(re_re_cropping_expansive_v2[0][3])
                                                        re_re_cropping_expansive_re.append(re_re_cropping_expansive_v2[0][2])
                                                    else:
                                                        re_re_cropping_expansive_re.append(re_re_cropping_expansive_v2[1][3])
                                                        re_re_cropping_expansive_re.append(re_re_cropping_expansive_v2[1][2])                                                            
                                                else:
                                                    # print("re_re_cropping[1]: ",re_re_cropping[1])
                                                    area_ori_points = (re_re_cropping[1][2],re_re_cropping[1][3],re_re[1][3],re_re[1][2])
                                                    area_ori_poly = Polygon(area_ori_points)
                                                    area_ori = area_ori_poly.area
                                                    intersection_ori = area_ori_poly.intersection(self.free_space_obstacles).area
                                                    re_re_cropping_expansive_v1 = self.Rectangle_generation(re_re_cropping[1][3][0], re_re_cropping[1][3][1],re_re[1][3][0],re_re[1][3][1],8)
                                                    if (((re_re_cropping[1][3][0]-re_re_cropping[1][2][0])*(re_re_cropping[1][3][0]-re_re_cropping_expansive_v1[0][3][0]))<=0) and (((re_re_cropping[1][3][1]-re_re_cropping[1][2][1])*(re_re_cropping[1][3][1]-re_re_cropping_expansive_v1[0][3][1]))<=0):
                                                        re_re_cropping_expansive_re = [re_re_cropping_expansive_v1[0][2],re_re_cropping_expansive_v1[0][3]]
                                                    else:
                                                        re_re_cropping_expansive_re = [re_re_cropping_expansive_v1[1][2],re_re_cropping_expansive_v1[1][3]]
                                                    
                                                    re_re_cropping_expansive_v2 = self.Rectangle_generation(re_re_cropping[1][2][0], re_re_cropping[1][2][1],re_re[1][2][0],re_re[1][2][1],8)
                                                    if (((re_re_cropping[1][2][0]-re_re_cropping[1][3][0])*(re_re_cropping[1][2][0]-re_re_cropping_expansive_v2[0][3][0]))<=0) and (((re_re_cropping[1][2][1]-re_re_cropping[1][3][1])*(re_re_cropping[1][2][1]-re_re_cropping_expansive_v2[0][3][1]))<=0):
                                                        re_re_cropping_expansive_re.append(re_re_cropping_expansive_v2[0][3])
                                                        re_re_cropping_expansive_re.append(re_re_cropping_expansive_v2[0][2])
                                                    else:
                                                        re_re_cropping_expansive_re.append(re_re_cropping_expansive_v2[1][3])
                                                        re_re_cropping_expansive_re.append(re_re_cropping_expansive_v2[1][2]) 
                                                # print("area_ori_points: ", area_ori_points)
                                                # print("re_re_cropping_expansive_re: ",re_re_cropping_expansive_re)
                                                re_re_cropping_expansive_re_poly = Polygon(re_re_cropping_expansive_re)
                                                if abs((re_re_cropping_expansive_re_poly.area - re_re_cropping_expansive_re_poly.intersection(self.free_space_obstacles).area) - (area_ori - intersection_ori)) > 3:
                                                    flag_sufficient_distance = False
                                                    # print("障碍物存在与边界的交点3")
                                                    # break
                                            
                                    
                                    # 判断条件四-停止条件：障碍物占据面积足够大
                                    if flag_sufficient_distance == True:
                                        area = self.segmentation_area_properties[index_test]["new_area"]
                                        if area < max((self.segmentation_area_properties[index_test]["origin_area"]*0.1), (1.5*self.config['meter2pixel']*self.config['meter2pixel'])):
                                            searching_times_2 = searching_times_2 + 10000
                                            searching_times_1 = searching_times_1 + 10000
                                            flag_sufficient_distance = False
                                            # print("因为面积原因停止探索")
                                            break
                                    



                                    
                                    # print("=== testing 3===") #2_11 test
                                    if flag_sufficient_distance == True:
                                        center_A_rect_obstacle_x = -1
                                        center_A_rect_obstacle_y = -1
                                        circle_A_a_size = -1
                                        circle_A_b_size = -1
                                        dx_p = possible_corner[0][0] - possible_corner[1][0]
                                        dy_p = possible_corner[0][1] - possible_corner[1][1]
                                        if dx_p == 0:
                                            kt_p = 90
                                        else:
                                            kt_p = int(round(math.atan(dy_p*1.0/dx_p)))
                                            if kt_p < 0:
                                                kt_p = math.pi + kt_p
                                            kt_p = 180/math.pi * kt_p
                                        if fake_flag_obstacle_A == False:
                                            if obs_type_A == 0:
                                                if A_other_direction == 1:
                                                    pass
                                                if obs_C_flag == True:
                                                    obstacle_possbile_shape_re = new_rect_c
                                            if obs_type_A == 1:
                                                center_A_rect_obstacle_x,center_A_rect_obstacle_y = self.cross_point_function((obstacle_possbile_shape_re[0][0],obstacle_possbile_shape_re[0][1],obstacle_possbile_shape_re[2][0],obstacle_possbile_shape_re[2][1]),(obstacle_possbile_shape_re[1][0],obstacle_possbile_shape_re[1][1],obstacle_possbile_shape_re[3][0],obstacle_possbile_shape_re[3][1]))
                                                circle_A_a_size = obs_type_A_a_size  / 2
                                                circle_A_b_size = obs_type_A_b_size  / 2
                                                


                                            else:
                                                if obs_type_A == 2:
                                                    if obs_C_flag == False:
                                                        A_polygon_vertex_re = [(obstacle_possbile_shape_re[0][0],obstacle_possbile_shape_re[0][1]),(obstacle_possbile_shape_re[1][0],obstacle_possbile_shape_re[1][1])]
                                                        line_2_points_coding_A = [0,1,2,3,0]
                                                        side_choice_vertex_A = [1]
                                                        A_vertex_generation = np.random.choice([2,3,4,5],p=[0.2,0.25,0.15,0.4]) 
                                                        
                                                        while (1):
                                                            side_choice_vertex_temp_single_A = int(np.random.randint(1,4))
                                                            if side_choice_vertex_temp_single_A == 3:
                                                                if side_choice_vertex_A.count(side_choice_vertex_temp_single_A) < 1:
                                                                    side_choice_vertex_A.append(side_choice_vertex_temp_single_A)
                                                                    # print(side_choice_vertex_A)
                                                            else:
                                                                if side_choice_vertex_A.count(side_choice_vertex_temp_single_A) < 2:
                                                                    side_choice_vertex_A.append(side_choice_vertex_temp_single_A)
                                                                    # print(side_choice_vertex_A)
                                                            if len(side_choice_vertex_A) == A_vertex_generation:
                                                                break
                                                        # print("obs_type_A == 2, A_vertex_generation: ",A_vertex_generation)
                                                        
                                                        last_ii_index_A = 0
                                                        side_choice_vertex_A.sort()
                                                        flag_point_selection_error_A = False
                                                        
                                                        for j_generation_index, j_generation_point in enumerate(side_choice_vertex_A):
                                                            if j_generation_index == 0:
                                                                last_ii_index_A = 0
                                                                continue
                                                            gggg_selection = True
                                                            num_shape_iteration_A = 0
                                                            
                                                            while gggg_selection and num_shape_iteration_A < 90:
                                                                num_shape_iteration_A = num_shape_iteration_A + 1
                                                                point_seg_A_0 = [obstacle_possbile_shape_re[line_2_points_coding_A[j_generation_point]][0],obstacle_possbile_shape_re[line_2_points_coding_A[j_generation_point]][1]]
                                                                point_line_x_A, point_line_y_A, point_line_ii_A = self.point_selection_line(obstacle_possbile_shape_re[line_2_points_coding_A[j_generation_point]][0],obstacle_possbile_shape_re[line_2_points_coding_A[j_generation_point]][1],obstacle_possbile_shape_re[line_2_points_coding_A[j_generation_point+1]][0],obstacle_possbile_shape_re[line_2_points_coding_A[j_generation_point+1]][1])                                                            
                                                                gggg_selection = False
                                                                if point_line_x_A < 0 or point_line_x_A < 0:
                                                                    flag_point_selection_error_A = True
                                                                    break
                                                                else:
                                                                    for last_point_A_vertex in A_polygon_vertex_re:
                                                                        if (point_line_x_A == last_point_A_vertex[0]) and (point_line_y_A == last_point_A_vertex[1]):
                                                                            gggg_selection = True
                                                                            break
                                                                    # if j_generation_point == side_choice_vertex_A[j_generation_index - 1]:
                                                                    #     if last_ii_index_A == point_line_ii_A:
                                                                    #         gggg_selection = True
                                                                    if j_generation_point > 0:
                                                                        if (side_choice_vertex_A.count(j_generation_point -1 ) > 1) and (point_line_x_A == point_seg_A_0[0] and point_line_y_A == point_seg_A_0[1]):
                                                                            gggg_selection = True
                                                            if gggg_selection == True:
                                                                flag_point_selection_error_A = True
                                                            
                                                            if flag_point_selection_error_A == False:
                                                                if j_generation_point == side_choice_vertex_A[j_generation_index-1]:
                                                                    if last_ii_index_A > point_line_ii_A:
                                                                        A_polygon_vertex_re.insert(-1,(point_line_x_A, point_line_y_A))
                                                                    else:
                                                                        A_polygon_vertex_re.append((point_line_x_A, point_line_y_A))
                                                                else:
                                                                    A_polygon_vertex_re.append((point_line_x_A, point_line_y_A))
                                                                last_ii_index_A = copy.deepcopy(point_line_ii_A)
                                                            else:
                                                                break
                                                        # print("A_polygon_vertex_re: ",A_polygon_vertex_re)
                                                        if flag_point_selection_error_A == False:
                                                            obstacle_possbile_shape_re = A_polygon_vertex_re
                                                    else:
                                                        polygon_vertex_re = []
                                                        line_2_points_coding = [0,1,2,3,0]
                                                        side_choice_vertex = [0]
                                                        polygon_vertex_generation = np.random.choice([3,4,5,6],p=[0.2,0.25,0.15,0.4]) 
                                                        
                                                        while (1):
                                                            side_choice_vertex_temp_single = int(np.random.randint(1,4))
                                                            if side_choice_vertex.count(side_choice_vertex_temp_single) < 2:
                                                                side_choice_vertex.append(side_choice_vertex_temp_single)
                                                            if len(side_choice_vertex) == polygon_vertex_generation:
                                                                break
                                                        last_ii_index = -1
                                                        side_choice_vertex.sort()
                                                        flag_point_selection_error = False
                                                        for i_triangle_index, i_triangle_point in enumerate(side_choice_vertex):
                                                            line_2_points_coding = [0,1,2,3,0]
                                                            hhhh_selection = True
                                                            num_shape_iteration_B = 0
                                                            while (hhhh_selection and num_shape_iteration_B < 90):
                                                                num_shape_iteration_B = num_shape_iteration_B + 1
                                                                point_seg_0 = [obstacle_possbile_shape_re[line_2_points_coding[i_triangle_point]][0],obstacle_possbile_shape_re[line_2_points_coding[i_triangle_point]][1]]
                                                                point_selection_line_x, point_selection_line_y, point_selection_line_ii = self.point_selection_line_v2_parameter(obstacle_possbile_shape_re[line_2_points_coding[i_triangle_point]][0],obstacle_possbile_shape_re[line_2_points_coding[i_triangle_point]][1],obstacle_possbile_shape_re[line_2_points_coding[i_triangle_point+1]][0],obstacle_possbile_shape_re[line_2_points_coding[i_triangle_point+1]][1],point_len=5)
                                                                hhhh_selection = False
                                                                if point_selection_line_x < 0 or point_selection_line_y < 0:
                                                                    flag_point_selection_error = True
                                                                else:
                                                                    for last_point_vertex_C in polygon_vertex_re:
                                                                        if (point_selection_line_x == last_point_vertex_C[0]) and (point_selection_line_y == last_point_vertex_C[1]):
                                                                            hhhh_selection = True
                                                                            break
                                                                    if i_triangle_point > 0:
                                                                        if (side_choice_vertex.count(i_triangle_point-1)> 1)  and (point_selection_line_x == point_seg_0[0] and point_selection_line_y == point_seg_0[1]):
                                                                            hhhh_selection = True
                                                                    else:
                                                                        if (side_choice_vertex.count(3)> 1)  and (point_selection_line_x == point_seg_0[0] and point_selection_line_y == point_seg_0[1]):
                                                                            hhhh_selection = True
                                                                    if i_triangle_point == 1 and (point_selection_line_x == point_seg_0[0] and point_selection_line_y == point_seg_0[1]):
                                                                        hhhh_selection = True
                                                            if flag_point_selection_error == False:
                                                                if (i_triangle_index > 0) and (i_triangle_point == side_choice_vertex[i_triangle_index-1]):
                                                                    if last_ii_index > point_selection_line_ii:
                                                                        polygon_vertex_re.insert(-1,(point_selection_line_x, point_selection_line_y))
                                                                    else:
                                                                        polygon_vertex_re.append((point_selection_line_x, point_selection_line_y))
                                                                else:
                                                                    polygon_vertex_re.append((point_selection_line_x, point_selection_line_y))
                                                                last_ii_index = copy.deepcopy(point_selection_line_ii)
                                                            else:
                                                                break
                                                        if flag_point_selection_error == False:
                                                            obstacle_possbile_shape_re = polygon_vertex_re
                                            area_A_obs = Polygon(obstacle_possbile_shape_re)
                                            if area_A_obs.is_valid == False:
                                                print("\033[43;34m area_A_obs的边界无效\033[0m")
                                                print("\033[43;34m{}\033[0m".format(obstacle_possbile_shape_re))
                                                find_ok = False
                                                continue
                                            editable_map = np.zeros_like(self.world)
                                            editable_map[self.world == self.map_color['obstacle']] = self.map_color['obstacle']
                                            editable_map[self.world == self.map_color['free']] = self.map_color['free']
                                            temp_obastacle_map = np.zeros_like(editable_map)
                                            # print()
                                
                                            # print("=== testing 4===") #2_11 test
                                            if obs_type_A == 1:
                                                if obs_C_flag == False:
                                                    if close_side_type == 1:
                                                        d_x_vertical = obstacle_possbile_shape_re[1][0] - obstacle_possbile_shape_re[2][0]
                                                        d_y_vertical = obstacle_possbile_shape_re[1][1] - obstacle_possbile_shape_re[2][1]
                                                        if d_x_vertical == 0:
                                                            ellipse_direction = 90
                                                        else:
                                                            ellipse_direction = int(round(math.atan(d_y_vertical*1.0/d_x_vertical)))
                                                            if ellipse_direction < 0:
                                                                ellipse_direction = math.pi + ellipse_direction
                                                            ellipse_direction = 180/math.pi * ellipse_direction
                                                    if close_side_type == 0:
                                                        ellipse_direction = kt_p
                                                    if close_side_type == 2:
                                                        ellipse_direction = kt_p
                                                    flag_rect_under = True
                                                    k_A_1,b_A_1 = self.line_range_function(obstacle_possbile_shape_re[1][0],obstacle_possbile_shape_re[1][1], obstacle_possbile_shape_re[2][0],obstacle_possbile_shape_re[2][1])
                                                    # print("obstacle_possbile_shape_re: ",obstacle_possbile_shape_re)
                                                    # print("k_A_1,b_A_1: ",k_A_1,b_A_1)
                                                    ii_point_A = int(np.random.randint(3,8))
                                                    k_A_2,b_A_2 = self.line_range_function(obstacle_possbile_shape_re[0][0],obstacle_possbile_shape_re[0][1], obstacle_possbile_shape_re[3][0],obstacle_possbile_shape_re[3][1])
                                                    # print("k_A_2,b_A_2: ",k_A_2,b_A_2)
                                                    # print("ii_point_A:",ii_point_A)
                                                    max_k_A_x_1 = max(obstacle_possbile_shape_re[1][0],obstacle_possbile_shape_re[2][0])
                                                    min_k_A_x_1 = min(obstacle_possbile_shape_re[1][0],obstacle_possbile_shape_re[2][0])
                                                    max_k_A_y_1 = max(obstacle_possbile_shape_re[1][1],obstacle_possbile_shape_re[2][1])
                                                    min_k_A_y_1 = min(obstacle_possbile_shape_re[1][1],obstacle_possbile_shape_re[2][1])
                                                    max_k_A_x_2 = max(obstacle_possbile_shape_re[0][0],obstacle_possbile_shape_re[3][0])
                                                    min_k_A_x_2 = min(obstacle_possbile_shape_re[0][0],obstacle_possbile_shape_re[3][0])
                                                    max_k_A_y_2 = max(obstacle_possbile_shape_re[0][1],obstacle_possbile_shape_re[3][1])
                                                    min_k_A_y_2 = min(obstacle_possbile_shape_re[0][1],obstacle_possbile_shape_re[3][1])
                                                    if k_A_1 ==None:
                                                        n_k_A_1 = (obstacle_possbile_shape_re[2][1] - obstacle_possbile_shape_re[1][1])*1.0/9
                                                        p_y_k_A_1 = obstacle_possbile_shape_re[1][1] + n_k_A_1 * ii_point_A
                                                        p_x_k_A_1 = obstacle_possbile_shape_re[1][0]
                                                        # print("n_k_A_1, p_y_k_A_1, p_x_k_A_1: ",n_k_A_1, p_y_k_A_1, p_x_k_A_1)

                                                        if (p_y_k_A_1 <= max_k_A_y_1) and (p_y_k_A_1 >= min_k_A_y_1):
                                                            pass
                                                        else:
                                                            flag_rect_under = False
                                                            print("1")
                                                    else:
                                                        n_k_A_1 = (obstacle_possbile_shape_re[2][0] - obstacle_possbile_shape_re[1][0])*1.0/9
                                                        p_x_k_A_1 = obstacle_possbile_shape_re[1][0] + n_k_A_1 * ii_point_A
                                                        p_y_k_A_1 = k_A_1*p_x_k_A_1 + b_A_1
                                                        # print("n_k_A_1, p_y_k_A_1, p_x_k_A_1: ",n_k_A_1, p_y_k_A_1, p_x_k_A_1)
                                                        if (p_y_k_A_1 <= max_k_A_y_1) and (p_y_k_A_1 >= min_k_A_y_1):
                                                            if (p_x_k_A_1 <= max_k_A_x_1) and (p_x_k_A_1 >= min_k_A_x_1):
                                                                pass
                                                            else:
                                                                flag_rect_under = False
                                                                print("2")
                                                        else:
                                                            flag_rect_under = False
                                                            print("3")

                                                    if k_A_2 ==None:
                                                        n_k_A_2 = (obstacle_possbile_shape_re[3][1] - obstacle_possbile_shape_re[0][1])*1.0/9
                                                        p_y_k_A_2 = obstacle_possbile_shape_re[0][1] + n_k_A_2 * ii_point_A
                                                        p_x_k_A_2 = obstacle_possbile_shape_re[0][0]
                                                        # print("n_k_A_2, p_y_k_A_2, p_x_k_A_2", n_k_A_2, p_y_k_A_2, p_x_k_A_2)
                                                        if (p_y_k_A_2 <= max_k_A_y_2) and (p_y_k_A_2 >= min_k_A_y_2):
                                                            pass
                                                        else:
                                                            flag_rect_under = False
                                                            print("4")

                                                    else:
                                                        n_k_A_2 = (obstacle_possbile_shape_re[3][0] - obstacle_possbile_shape_re[0][0])*1.0/9
                                                        p_x_k_A_2 = obstacle_possbile_shape_re[0][0] + n_k_A_2 * ii_point_A
                                                        p_y_k_A_2 = k_A_2*p_x_k_A_2 + b_A_2
                                                        # print("n_k_A_2, p_y_k_A_2, p_x_k_A_2", n_k_A_2, p_y_k_A_2, p_x_k_A_2)
                                                        
                                                        if (p_y_k_A_2 <= max_k_A_y_2) and (p_y_k_A_2 >= min_k_A_y_2):
                                                            if (p_x_k_A_2 <= max_k_A_x_2) and (p_x_k_A_2 >= min_k_A_x_2):
                                                                pass
                                                            else:
                                                                flag_rect_under = False
                                                                print("5")
                                                        else:
                                                            flag_rect_under = False
                                                            print("6")
                                                    if flag_rect_under == True:
                                                        # print("(center_A_rect_obstacle_x,center_A_rect_obstacle_y): ",(center_A_rect_obstacle_x,center_A_rect_obstacle_y))
                                                        draw_iscretization_no = False
                                                        if flag_2d_laser_lattice == 1:
                                                            draw_iscretization_no = self.draw_graphic_iscretization(obs_type_A, [int(round(center_A_rect_obstacle_x)), int(round(center_A_rect_obstacle_y)), int(round(circle_A_a_size)), int(round(circle_A_b_size)), obstacle_possbile_shape_re], kt_p, temp_obastacle_map)
                                                        if flag_2d_laser_lattice == 0 or draw_iscretization_no == True:
                                                            cv2.ellipse(temp_obastacle_map, (int(round(center_A_rect_obstacle_x)),int(round(center_A_rect_obstacle_y))), (int(round(circle_A_a_size)),int(round(circle_A_b_size))), ellipse_direction, 0,360, self.map_color["obstacle"],thickness=-1)
                                                            cv2.fillPoly(temp_obastacle_map, [np.array([obstacle_possbile_shape_re[0],obstacle_possbile_shape_re[1],(p_x_k_A_1,p_y_k_A_1),(p_x_k_A_2,p_y_k_A_2)],np.int32)], self.map_color["obstacle"])

                                                    else:
                                                        print("bei pian le")

                                                        draw_iscretization_no = False
                                                        if flag_2d_laser_lattice == 1:
                                                            draw_iscretization_no = self.draw_graphic_iscretization(obs_type_A, obstacle_possbile_shape_re, kt_p, temp_obastacle_map)
                                                        if flag_2d_laser_lattice == 0 or draw_iscretization_no == True:
                                                            cv2.fillPoly(temp_obastacle_map, [np.array(obstacle_possbile_shape_re,np.int32)], self.map_color["obstacle"])
                                                else:
                                                    if ellipse_c_pos != []:
                                                        if (((close_side_type == 0) or (close_side_type == 2)) and ellipse_c_pos[2] == 1) or (close_side_type == 1 and ellipse_c_pos[2] == 0):
                                                            x_new_rect_c_11, y_new_rect_c_11 = self.line_segmentation_length(obstacle_possbile_shape_re[0][0],obstacle_possbile_shape_re[0][1],obstacle_possbile_shape_re[1][0],obstacle_possbile_shape_re[1][1],ellipse_c_pos[0])
                                                            x_new_rect_c_12, y_new_rect_c_12 = self.line_segmentation_length(obstacle_possbile_shape_re[3][0],obstacle_possbile_shape_re[3][1],obstacle_possbile_shape_re[2][0],obstacle_possbile_shape_re[2][1],ellipse_c_pos[0])
                                                            x_new_rect_c_21, y_new_rect_c_21 = self.line_segmentation_length(obstacle_possbile_shape_re[0][0],obstacle_possbile_shape_re[0][1],obstacle_possbile_shape_re[3][0],obstacle_possbile_shape_re[3][1],ellipse_c_pos[1])
                                                            x_new_rect_c_22, y_new_rect_c_22 = self.line_segmentation_length(obstacle_possbile_shape_re[1][0],obstacle_possbile_shape_re[1][1],obstacle_possbile_shape_re[2][0],obstacle_possbile_shape_re[2][1],ellipse_c_pos[1])
                                                            center_x_new_rect_c,center_y_new_rect_c = self.cross_point_function((x_new_rect_c_11, y_new_rect_c_11,x_new_rect_c_12, y_new_rect_c_12),(x_new_rect_c_21, y_new_rect_c_21,x_new_rect_c_22, y_new_rect_c_22))
                                                            theta_ttt = math.atan2((obstacle_possbile_shape_re[1][1]-obstacle_possbile_shape_re[0][1]), (obstacle_possbile_shape_re[1][0]-obstacle_possbile_shape_re[0][0]))
                                                            theta_ttt_u = math.atan2((obstacle_possbile_shape_re[3][1]-obstacle_possbile_shape_re[0][1]), (obstacle_possbile_shape_re[3][0]-obstacle_possbile_shape_re[0][0]))
                                                            # if theta_ttt_u - theta_ttt > 0:
                                                            #     theta_ellipse_c_p = theta_ttt + ellipse_c_pos[4]
                                                            # else:
                                                            #     theta_ellipse_c_p = theta_ttt - ellipse_c_pos[4]
                                                            # if theta_ellipse_c_p > math.pi:
                                                            #     theta_ellipse_c_p = theta_ellipse_c_p - 2*math.pi
                                                            # if theta_ellipse_c_p < -math.pi:
                                                            #     theta_ellipse_c_p = theta_ellipse_c_p + 2*math.pi
                                                            # theta_ellipse_c_p = 180/math.pi * theta_ellipse_c_p
                                                            # while theta_ellipse_c_p < 0:
                                                            #     theta_ellipse_c_p = theta_ellipse_c_p + 360
                                                            # while theta_ellipse_c_p > 360:
                                                            #     theta_ellipse_c_p = theta_ellipse_c_p - 360
                                                            # cv2.ellipse(temp_obastacle_map, (int(round(center_x_new_rect_c)),int(round(center_y_new_rect_c))), (int(round(ellipse_c_pos[3][0])),int(round(ellipse_c_pos[3][1]))), theta_ellipse_c_p, 0,360, self.map_color["obstacle"],thickness=-1)
                                                        else:
                                                            x_new_rect_c_11, y_new_rect_c_11 = self.line_segmentation_length(obstacle_possbile_shape_re[1][0],obstacle_possbile_shape_re[1][1],obstacle_possbile_shape_re[0][0],obstacle_possbile_shape_re[0][1],ellipse_c_pos[1])
                                                            x_new_rect_c_12, y_new_rect_c_12 = self.line_segmentation_length(obstacle_possbile_shape_re[2][0],obstacle_possbile_shape_re[2][1],obstacle_possbile_shape_re[3][0],obstacle_possbile_shape_re[3][1],ellipse_c_pos[1])
                                                            x_new_rect_c_21, y_new_rect_c_21 = self.line_segmentation_length(obstacle_possbile_shape_re[1][0],obstacle_possbile_shape_re[1][1],obstacle_possbile_shape_re[2][0],obstacle_possbile_shape_re[2][1],ellipse_c_pos[0])
                                                            x_new_rect_c_22, y_new_rect_c_22 = self.line_segmentation_length(obstacle_possbile_shape_re[0][0],obstacle_possbile_shape_re[0][1],obstacle_possbile_shape_re[3][0],obstacle_possbile_shape_re[3][1],ellipse_c_pos[0])
                                                            center_x_new_rect_c,center_y_new_rect_c = self.cross_point_function((x_new_rect_c_11, y_new_rect_c_11,x_new_rect_c_12, y_new_rect_c_12),(x_new_rect_c_21, y_new_rect_c_21,x_new_rect_c_22, y_new_rect_c_22))
                                                            theta_ttt = math.atan2((obstacle_possbile_shape_re[2][1]-obstacle_possbile_shape_re[1][1]), (obstacle_possbile_shape_re[2][0]-obstacle_possbile_shape_re[1][0]))
                                                            theta_ttt_u = math.atan2((obstacle_possbile_shape_re[0][1]-obstacle_possbile_shape_re[1][1]), (obstacle_possbile_shape_re[0][0]-obstacle_possbile_shape_re[1][0]))
                                                        delta_theta_ttt = theta_ttt_u - theta_ttt
                                                        if  abs(delta_theta_ttt) > 0.6*math.pi:
                                                            if delta_theta_ttt < 0:
                                                                delta_theta_ttt = delta_theta_ttt + 2*math.pi
                                                            else:
                                                                delta_theta_ttt = delta_theta_ttt - 2*math.pi
                                                            
                                                        if delta_theta_ttt > 0:
                                                            theta_ellipse_c_p = theta_ttt + (math.pi / 180 * ellipse_c_pos[4])
                                                        else:
                                                            theta_ellipse_c_p = theta_ttt - (math.pi / 180 * ellipse_c_pos[4])
                                                        while theta_ellipse_c_p > math.pi:
                                                            theta_ellipse_c_p = theta_ellipse_c_p - 2*math.pi
                                                        while theta_ellipse_c_p <= -math.pi:
                                                            theta_ellipse_c_p = theta_ellipse_c_p + 2*math.pi
                                                        theta_ellipse_c_p = 180/math.pi * theta_ellipse_c_p
                                                        while theta_ellipse_c_p < 0:
                                                            theta_ellipse_c_p = theta_ellipse_c_p + 360
                                                        while theta_ellipse_c_p > 360:
                                                            theta_ellipse_c_p = theta_ellipse_c_p - 360
                                                        draw_iscretization_no = False
                                                        if flag_2d_laser_lattice == 1:
                                                            cthe_b = math.cos(np.pi/180* theta_ellipse_c_p)
                                                            sthe_b = math.sin(np.pi/180* theta_ellipse_c_p)
                                                            rect_b_cc = [[int(round(center_x_new_rect_c + (-ellipse_c_pos[3][0] * cthe_b - -ellipse_c_pos[3][1] * sthe_b))), int(round(center_y_new_rect_c + (-ellipse_c_pos[3][0] * sthe_b + -ellipse_c_pos[3][1] * cthe_b)))],
                                                                            [int(round(center_x_new_rect_c + (-ellipse_c_pos[3][0] * cthe_b - ellipse_c_pos[3][1] * sthe_b))), int(round(center_y_new_rect_c + (-ellipse_c_pos[3][0] * sthe_b + ellipse_c_pos[3][1] * cthe_b)))],
                                                                            [int(round(center_x_new_rect_c + (ellipse_c_pos[3][0] * cthe_b - ellipse_c_pos[3][1] * sthe_b))), int(round(center_y_new_rect_c + (ellipse_c_pos[3][0] * sthe_b + ellipse_c_pos[3][1] * cthe_b)))],
                                                                            [int(round(center_x_new_rect_c + (ellipse_c_pos[3][0] * cthe_b - -ellipse_c_pos[3][1] * sthe_b))), int(round(center_y_new_rect_c + (ellipse_c_pos[3][0] * sthe_b + -ellipse_c_pos[3][1] * cthe_b)))]]
                                                            draw_iscretization_no = self.draw_graphic_iscretization(1, [int(round(center_x_new_rect_c)), int(round(center_y_new_rect_c)), int(round(ellipse_c_pos[3][0])), int(round(ellipse_c_pos[3][1])), rect_b_cc], theta_ellipse_c_p, temp_obastacle_map)
                                                        if flag_2d_laser_lattice == 0 or draw_iscretization_no == True:
                                                            cv2.ellipse(temp_obastacle_map, (int(round(center_x_new_rect_c)),int(round(center_y_new_rect_c))), (int(round(ellipse_c_pos[3][0])),int(round(ellipse_c_pos[3][1]))), theta_ellipse_c_p, 0,360, self.map_color["obstacle"],thickness=-1)
                                                            # cv2.ellipse(temp_obastacle_map, (int(round(center_x_new_rect_c)),int(round(center_y_new_rect_c))), (int(round(ellipse_c_pos[3][0])),int(round(ellipse_c_pos[3][1]))), theta_ellipse_c_p, 0,360, self.map_color["obstacle"],thickness=-1)
                                            else:
                                                draw_iscretization_no = False
                                                if flag_2d_laser_lattice == 1:
                                                    draw_iscretization_no = self.draw_graphic_iscretization(obs_type_A, obstacle_possbile_shape_re, kt_p, temp_obastacle_map)
                                                if flag_2d_laser_lattice == 0 or draw_iscretization_no == True:
                                                    cv2.fillPoly(temp_obastacle_map, [np.array(obstacle_possbile_shape_re,np.int32)], self.map_color["obstacle"])
                                                    
                                            # print("3")
                                            # cv2.fillPoly(temp_obastacle_map, [np.array(obstacle_possbile_shape_re,np.int32)], self.map_color["obstacle"])
                                            editable_map[temp_obastacle_map==0] = self.map_color['obstacle']
                                            # temp_obastacle_map[editable_map==self.map_color['free']] = 145
                                            # temp_obastacle_map[editable_map==self.map_color['obstacle']] = 145
                                            # print("shape",temp_obastacle_map.shape, type(temp_obastacle_map),editable_map.shape, type(editable_map) )
                                            self.world[editable_map==self.map_color['free']] = self.map_color['obstacle']
                                            added_obstacles_list.append(copy.deepcopy(obstacle_possbile_shape_re))
                                            self.segmentation_area_properties[index_test]["new_area"] = self.segmentation_area_properties[index_test]["new_area"] - (Polygon(obstacle_possbile_shape_re).area)
                                            
                                            # print("---随机到的障碍物起始位置: ",close_side_corner_begin,"对应索引(kk1,kk2,begin_kk1,begin_kk2,end_kk1,end_kk2,points_test_segmentation[boundary_free_list[begin_kk1][begin_kk2]],points_test_segmentation[boundary_free_list[end_kk1][end_kk2]]): ",kk1,kk2,begin_kk1,begin_kk2,end_kk1,end_kk2,points_test_segmentation[boundary_free_list[begin_kk1][begin_kk2]],points_test_segmentation[boundary_free_list[end_kk1][end_kk2]])
                                            # print("---障碍物结束位置: ",possible_corner)
                                            # print("上次边界点列表: ", boundary_free_list)


                                            # boundary_append_obstacle_points_all = {}
                                            if obs_C_flag == False:
                                                if flag_2d_laser_lattice == 0 or draw_iscretization_no == True:
                                                    # if obs_type_A == 0 or obs_type_A == 2:
                                                    #     tmp_boundary_append_obstacle_points = []
                                                    #     for i_replacement_index in range(len(obstacle_possbile_shape_re)):
                                                    #         if i_replacement_index == 0:
                                                    #             tmp_boundary_append_obstacle_points.append(obstacle_possbile_shape_re[i_replacement_index])
                                                    #         else:
                                                    #             tmp_boundary_append_obstacle_points.append(obstacle_possbile_shape_re[len(obstacle_possbile_shape_re)-i_replacement_index])
                                                    #     boundary_append_obstacle_points.update({boundary_append_obstacle_points_count:tmp_boundary_append_obstacle_points})
                                                    #     for Replacement_index in range(begin_kk2, end_kk2+1):
                                                    #         boundary_append_obstacle_points_list[boundary_free_list[begin_kk1][Replacement_index]] = boundary_append_obstacle_points_count
                                                    #     boundary_append_obstacle_points_count = boundary_append_obstacle_points_count - 1
                                                    # if obs_type_A == 1:
                                                    if True:
                                                        # if flag_rect_under == True:
                                                        if True:
                                                            temp_obastacle_map_2 = np.zeros_like(temp_obastacle_map,np.uint8)
                                                            temp_obastacle_map_2[temp_obastacle_map ==  self.map_color["obstacle"]] = 255
                                                            contours_A1, hierarchy_A1 = cv2.findContours(temp_obastacle_map_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                                                            min_obstacle_0_dis = 100000000
                                                            min_obstacle_0_index = 0
                                                            min_obstacle_1_dis = 100000000
                                                            min_obstacle_1_index = 0
                                                            # cv2.imshow('contour', temp_obastacle_map_2)
                                                            # cv2.waitKey(0)
                                                            # print("contours_A1,",contours_A1)
                                                            contours_A1 = contours_A1[0]
                                                            if (contours_A1[0][0][0] == contours_A1[-1][0][0]) and (contours_A1[0][0][1] == contours_A1[-1][0][1]):
                                                                contours_A1 = contours_A1[:-1]
                                                            for i_A1_index, i_A1 in enumerate(contours_A1):
                                                                i_A1 = i_A1[0]
                                                                a0_obstacle_0_dis = math.sqrt((i_A1[0] - obstacle_possbile_shape_re[0][0])*(i_A1[0] - obstacle_possbile_shape_re[0][0])+(i_A1[1] - obstacle_possbile_shape_re[0][1])*(i_A1[1] - obstacle_possbile_shape_re[0][1]))
                                                                a1_obstacle_1_dis = math.sqrt((i_A1[0] - obstacle_possbile_shape_re[1][0])*(i_A1[0] - obstacle_possbile_shape_re[1][0])+(i_A1[1] - obstacle_possbile_shape_re[1][1])*(i_A1[1] - obstacle_possbile_shape_re[1][1]))
                                                                if a0_obstacle_0_dis < min_obstacle_0_dis:
                                                                    min_obstacle_0_dis = a0_obstacle_0_dis
                                                                    min_obstacle_0_index = copy.deepcopy(i_A1_index)
                                                                if a1_obstacle_1_dis < min_obstacle_1_dis:
                                                                    min_obstacle_1_dis = a1_obstacle_1_dis
                                                                    min_obstacle_1_index = copy.deepcopy(i_A1_index)
                                                            if (min_obstacle_0_index <= min_obstacle_1_index):
                                                                route_1 = list(range(min_obstacle_0_index,min_obstacle_1_index + 1))
                                                                route_2 = list(range(min_obstacle_0_index,-1,-1)) + list(range(len(contours_A1)-1, min_obstacle_1_index -1, -1))
                                                            else:
                                                                route_1 = list(range(min_obstacle_0_index,min_obstacle_1_index - 1,-1))
                                                                route_2 = list(range(min_obstacle_0_index, len(contours_A1))) + list(range(0,min_obstacle_1_index))
                                                            route_1_distance = 0
                                                            for u_a1 in range(len(route_1)):
                                                                if u_a1 < (len(route_1) -1):
                                                                    route_1_distance = route_1_distance + math.sqrt((contours_A1[route_1[u_a1]][0][0] - contours_A1[route_1[u_a1+1]][0][0])*(contours_A1[route_1[u_a1]][0][0] - contours_A1[route_1[u_a1+1]][0][0])+(contours_A1[route_1[u_a1]][0][1] - contours_A1[route_1[u_a1+1]][0][1])*(contours_A1[route_1[u_a1]][0][1] - contours_A1[route_1[u_a1+1]][0][1]))
                                                            route_2_distance = 0
                                                            for u_a2 in range(len(route_2)):
                                                                if u_a2 < (len(route_2) -1):
                                                                    route_2_distance = route_2_distance + math.sqrt((contours_A1[route_2[u_a2]][0][0] - contours_A1[route_2[u_a2+1]][0][0])*(contours_A1[route_2[u_a2]][0][0] - contours_A1[route_2[u_a2+1]][0][0])+(contours_A1[route_2[u_a2]][0][1] - contours_A1[route_2[u_a2+1]][0][1])*(contours_A1[route_2[u_a2]][0][1] - contours_A1[route_2[u_a2+1]][0][1]))
                                                            tmp_boundary_append_obstacle_points = []
                                                            tmp_boundary_append_obstacle_points.append(obstacle_possbile_shape_re[0])
                                                            if route_1_distance <= route_2_distance:
                                                               for hh_ii in route_2[1:-1]:
                                                                   tmp_boundary_append_obstacle_points.append(contours_A1[hh_ii][0]) 
                                                            else:
                                                               for hh_ii in route_1[1:-1]:
                                                                   tmp_boundary_append_obstacle_points.append(contours_A1[hh_ii][0]) 
                                                            tmp_boundary_append_obstacle_points.append(obstacle_possbile_shape_re[1])
                                                            boundary_append_obstacle_points.update({boundary_append_obstacle_points_count:tmp_boundary_append_obstacle_points})
                                                            for Replacement_index in range(begin_kk2, end_kk2+1):
                                                                boundary_append_obstacle_points_list[boundary_free_list[begin_kk1][Replacement_index]] = boundary_append_obstacle_points_count
                                                            boundary_append_obstacle_points_count = boundary_append_obstacle_points_count - 1
                                                            
                                                                

                                                            


                                                    # else:  #obs_type_A == 1
                                            else:
                                                if flag_2d_laser_lattice == 0 or draw_iscretization_no == True:
                                                    temp_obastacle_map_3 = np.zeros_like(temp_obastacle_map,np.uint8)
                                                    temp_obastacle_map_3[temp_obastacle_map ==  self.map_color["obstacle"]] = 255
                                                    contours_A3, hierarchy_A1 = cv2.findContours(temp_obastacle_map_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                                                    contours_A3 = contours_A3[0]
                                                    conto_ttt = []
                                                    for A3_p in contours_A3:
                                                        A3_p = A3_p[0]
                                                        conto_ttt.append([int(A3_p[0]), int(A3_p[1])])
                                                    if len(conto_ttt) > 4:
                                                        if conto_ttt[-1] != conto_ttt[0]:
                                                            conto_ttt.append(conto_ttt[0])
                                                        B_contours.append(conto_ttt)




                                            temp_new_boundary_free_list = []
                                            for i_del in range(len(boundary_free_list)):
                                                if i_del == begin_kk1:
                                                    if len(range(begin_kk2,end_kk2+1)) < len(boundary_free_list[begin_kk1]):
                                                        temp_new_a_boundary = []
                                                        if begin_kk2 > 1:
                                                            temp_new_a_boundary.append(boundary_free_list[begin_kk1][0:begin_kk2])
                                                        if end_kk2 < (len(boundary_free_list[begin_kk1])-2):
                                                            temp_new_a_boundary.append(boundary_free_list[begin_kk1][(end_kk2+1):len(boundary_free_list[begin_kk1])])
                                                    else:
                                                        temp_new_a_boundary = []
                                                    for i_del_append in range(len(temp_new_a_boundary)):
                                                        temp_new_boundary_free_list.append(temp_new_a_boundary[i_del_append])
                                                else:
                                                    temp_new_boundary_free_list.append(boundary_free_list[i_del])
                                            boundary_free_list = temp_new_boundary_free_list
                                            # print("最新边界点列表: ", boundary_free_list)
                                            





                                                

                                            
                                            
                                            # print("---房间面积:",self.segmentation_area_properties[index_test]["new_area"])
                                            
                                            # print("生成障碍物形状: ",{0:"矩形",1:"椭圆",2:"多边形"}[obs_type_A])
                                            
                                            # print("---边长a的值:",obs_type_A_a_size)
                                            # print("---边长b的值:",obs_type_A_b_size)
                                            # print("---面积占比:",obs_type_A_a_size*obs_type_A_b_size/self.segmentation_area_properties[index_test]["new_area"])
                                            # print("---选择障碍物贴的贴边位置(0-长边贴,1短边贴,2两边贴):",close_side_type)
                                            
                                            
                                            # # print("当前距离值: ",dis_end_corner_tmp)
                                            # print("---选取的贴合墙的顶点: ",possible_corner)
                                            # print("---最终生成的多边形障碍物顶点: ",obstacle_possbile_shape_re)

                                            # print("原始顶点数据: ",re_re)
                                            # plt.clf()
                                            # plt.imshow(self.world,cmap="gray")
                                            # plt.draw()
                                            # plt.pause(0.01)
                                        else:
                                            pass
                                            # print("计算房间方向，不生成贴墙障碍物!!!")

                                        #记录房间主要方向
                                        if kt_p not in main_directions:
                                            main_directions.append(kt_p)
                                        



                                        # print("re: ",[np.array(list(area_re_k.exterior.coords),np.int32)])
                                        #一是把障碍物加到房间的空白区域，二是计算新的边界
                                        # if close_side_corner_begin > 
                                        # boundary_free_list[kk1] = 
                                        # possible_corner[0][0], possible_corner[0][1]
                                        find_ok = True
                                        # print("开始在长度为{}边界段中寻找障碍物结束位置".format(index_for_corner_range))
                                        break
                        # print("小循环结束次数: ",i_segta)
                        if find_ok == False:
                            searching_times_2 = searching_times_2 + 1
                            type_set.append(close_side_type)
                        else:
                            searching_times_2 = searching_times_2 + 10000
                            searching_times_1 = searching_times_1 + 10000
                    searching_times_1 = searching_times_1 + 1
                # self.world= world_obs.copy()
                # return
            # print("=== testing 5===") #2_11 test
            end_time2 = time.time()
            elapsed_time3 = end_time2 - building_start_time
            elapsed_time4 = end_time2 - building_ll_time
            print("Time-IN--A&C: {:.2f}-{:.2f}".format(elapsed_time4, elapsed_time3))
            building_ll_time = time.time()            
            print("--- B类型障碍物生成 ", obstacle_B_num)
                # input("等待继续...")

            
            point_num_obstacles_b = 0
            bbox = self.free_space_obstacles.bounds
            min_x,min_y,max_x,max_y = bbox[0],bbox[1],bbox[2],bbox[3]
            for ____ in range(obstacle_B_num):
                point_num_obstacles_b = point_num_obstacles_b + 1
                # print("\n$$ {}/{} $$".format(point_num_obstacles_b,obstacle_B_num))
                # input("等待继续...")
                obs_type_B = np.random.choice([0,1,2],p=[0.5,0.2,0.3])  #0矩形1椭圆2多边形
                # obs_type_B=0 #test
                searching_times_1_b = 0
                B_dense_flag = np.random.choice([0,1],p=[0.9, 0.1])
                B_direction_flag = np.random.choice([0,1],p=[0.7, 0.3])
                if sensor_type == 0:
                    flag_2d_laser_lattice = np.random.choice([0,1],p=[0.4,0.6])
                else:
                    flag_2d_laser_lattice = 0

                flag_b_ok = False
                # print("=== testing 6===") #2_11 test
                while (searching_times_1_b < 15) and (flag_b_ok == False):
                    # print("searching_times_1_b: ",searching_times_1_b)
                    obs_type_B_a_size, obs_type_B_b_size= self.random_multi_obstacle_size(0, obs_type_B, obs_sizeRange_room)
                    area = self.segmentation_area_properties[index_test]["new_area"]
                    area_limit = min((area*0.7), area-(1.5*self.config['meter2pixel']*self.config['meter2pixel']))
                    obs_type_B_area = obs_type_B_a_size * obs_type_B_b_size
                    cout_area_limit_searching_b = 1
                    go_next_flag_b = True
                    while (obs_type_B_area > area_limit) and cout_area_limit_searching_b < 251:
                        print("**重新确定尺寸**")
                        if cout_area_limit_searching_b % 50 == 0:
                            obs_sizeRange_room_ttt = []
                            length_limit = length_limit * 0.8
                            for seg_range in obs_sizeRange_room:
                                if length_limit < seg_range[1]:
                                    if length_limit <= seg_range[0]:
                                        pass
                                    else:
                                        obs_sizeRange_room_ttt.append([seg_range[0], area_limit])
                                else:
                                    obs_sizeRange_room_ttt.append(seg_range)
                            
                            obs_sizeRange_room = obs_sizeRange_room_ttt
                        obs_type_B_a_size, obs_type_B_b_size= self.random_multi_obstacle_size(0, obs_type_B, obs_sizeRange_room)
                        obs_type_B_area = obs_type_B_a_size * obs_type_B_b_size

                        # print("新尺寸: ", obs_sizeRange_room)
                            
                        cout_area_limit_searching_b = cout_area_limit_searching_b + 1
                        if len(obs_sizeRange_room) == 0:
                            go_next_flag_b = False

                    if cout_area_limit_searching_b == 251:
                        go_next_flag_b = False
                    if go_next_flag_b == False:
                        break
                    obs_type_B_a_size = obs_type_B_a_size/2
                    obs_type_B_b_size = obs_type_B_b_size/2
                    i_B_count = 0
                    # print("=== testing 7===") #2_11 test
                    while (i_B_count <100) and (flag_b_ok == False):
                        obs_y_b = int(np.random.randint(min_y,max_y))
                        obs_x_b = int(np.random.randint(min_x,max_x))
                        obstacle_B_theta = np.random.random()*360
                        if len(main_directions) > 0:
                            if B_direction_flag == 0:
                                __index__main_directions = int(np.random.randint(0,len(main_directions)))
                                obstacle_B_theta = main_directions[__index__main_directions]
                        cthe_b = math.cos(np.pi/180* obstacle_B_theta)
                        sthe_b = math.sin(np.pi/180* obstacle_B_theta)
                        # rect_b = [[obs_x_b + (-obs_type_B_a_size * cthe_b - -obs_type_B_b_size * sthe_b), obs_y_b + (-obs_type_B_a_size * sthe_b + -obs_type_B_b_size * cthe_b)],
                        #                 [obs_x_b + (-obs_type_B_a_size * cthe_b - obs_type_B_b_size * sthe_b), obs_y_b + (-obs_type_B_a_size * sthe_b + obs_type_B_b_size * cthe_b)],
                        #                 [obs_x_b + (obs_type_B_a_size * cthe_b - obs_type_B_b_size * sthe_b), obs_y_b + (obs_type_B_a_size * sthe_b + obs_type_B_b_size * cthe_b)],
                        #                 [obs_x_b + (obs_type_B_a_size * cthe_b - -obs_type_B_b_size * sthe_b), obs_y_b + (obs_type_B_a_size * sthe_b + -obs_type_B_b_size * cthe_b)]]
                        rect_b = [[int(round(obs_x_b + (-obs_type_B_a_size * cthe_b - -obs_type_B_b_size * sthe_b))), int(round(obs_y_b + (-obs_type_B_a_size * sthe_b + -obs_type_B_b_size * cthe_b)))],
                                        [int(round(obs_x_b + (-obs_type_B_a_size * cthe_b - obs_type_B_b_size * sthe_b))), int(round(obs_y_b + (-obs_type_B_a_size * sthe_b + obs_type_B_b_size * cthe_b)))],
                                        [int(round(obs_x_b + (obs_type_B_a_size * cthe_b - obs_type_B_b_size * sthe_b))), int(round(obs_y_b + (obs_type_B_a_size * sthe_b + obs_type_B_b_size * cthe_b)))],
                                        [int(round(obs_x_b + (obs_type_B_a_size * cthe_b - -obs_type_B_b_size * sthe_b))), int(round(obs_y_b + (obs_type_B_a_size * sthe_b + -obs_type_B_b_size * cthe_b)))]]
                        

                        rect_b_first_test_poly = Polygon(rect_b)
                        if rect_b_first_test_poly.is_valid == False:
                            print("\033[43;34m rect_b外接矩形无效\033[0m")
                            print(rect_b)
                            # flag_B_obstacle = False
                        # # #验证
                        # B_side_1_test = math.sqrt((rect_b[0][0]-rect_b[1][0])*(rect_b[0][0]-rect_b[1][0])+(rect_b[0][1]-rect_b[1][1])*(rect_b[0][1]-rect_b[1][1]))
                        # B_side_2_test = math.sqrt((rect_b[2][0]-rect_b[1][0])*(rect_b[2][0]-rect_b[1][0])+(rect_b[2][1]-rect_b[1][1])*(rect_b[2][1]-rect_b[1][1]))
                        # if ((max(B_side_1_test,B_side_2_test) - obs_type_B_a_size*2) < 0.00001) and ((min(B_side_1_test,B_side_2_test) - obs_type_B_b_size*2) < 0.00001):
                        #     center_x_test, center_y_test = self.cross_point_function([rect_b[0][0],rect_b[0][1],rect_b[2][0],rect_b[2][1]],[rect_b[1][0],rect_b[1][1],rect_b[3][0],rect_b[3][1]])
                        #     if ((center_x_test - obs_x_b) < 0.00001) and ((center_y_test - obs_y_b) < 0.00001):
                        #         # print("独立障碍物的外接矩形顶点计算成功!!!")
                        #         pass
                        #     else:
                        #         print("外接矩形计算失败,跳转至下一次循环!!!")
                        #         continue
                        # else:
                        #     print("矩形计算失败,跳转至下一次循环!!!")
                        #     continue
                            # 判断条件一：避免与墙距离太近堵住路
                        

                        # 1/2处障碍物形状针对性处理-外接多边形顶点计算
                        if obs_type_B == 2:
                            polygon_vertex_re = []
                            
                            polygon_vertex_generation = np.random.choice([3,4,5,6,7,8],p=[0.25,0.2,0.15,0.15,0.15,0.1])
                            side_choice_vertex = []
                            while (1):
                                side_choice_vertex_temp_single = int(np.random.randint(0,4))
                                if side_choice_vertex.count(side_choice_vertex_temp_single) < 2:
                                    side_choice_vertex.append(side_choice_vertex_temp_single)
                                if len(side_choice_vertex) == polygon_vertex_generation:
                                    break
                            last_ii_index = -1
                            side_choice_vertex.sort()
                            # print("test888: ", side_choice_vertex)
                            flag_point_selection_error = False
                            for i_triangle_index, i_triangle_point in enumerate(side_choice_vertex):
                                num_shape_iteration = 0
                                line_2_points_coding = [0,1,2,3,0]
                                hhhh_selection = True
                                while (hhhh_selection and num_shape_iteration <90):
                                    num_shape_iteration = num_shape_iteration + 1 
                                    point_seg_0 = [rect_b[line_2_points_coding[i_triangle_point]][0],rect_b[line_2_points_coding[i_triangle_point]][1]]
                                    point_selection_line_x, point_selection_line_y, point_selection_line_ii = self.point_selection_line(rect_b[line_2_points_coding[i_triangle_point]][0],rect_b[line_2_points_coding[i_triangle_point]][1],rect_b[line_2_points_coding[i_triangle_point+1]][0],rect_b[line_2_points_coding[i_triangle_point+1]][1])
                                    hhhh_selection = False
                                    if point_selection_line_x < 0 or point_selection_line_y < 0:
                                        flag_point_selection_error = True
                                        break
                                    else:                           
                                        for last_ppint_line in polygon_vertex_re:
                                            if (point_selection_line_x == last_ppint_line[0]) and (point_selection_line_y == last_ppint_line[1]):
                                                hhhh_selection = True
                                                # print("点重复")
                                                break
                                        if i_triangle_point > 0:
                                            if (side_choice_vertex.count(i_triangle_point-1)> 1)  and (point_selection_line_x == point_seg_0[0] and point_selection_line_y == point_seg_0[1]):
                                                hhhh_selection = True
                                        else:
                                            if (side_choice_vertex.count(3)> 1)  and (point_selection_line_x == point_seg_0[0] and point_selection_line_y == point_seg_0[1]):
                                                hhhh_selection = True
                                if hhhh_selection == True:
                                    flag_point_selection_error = True

                                if flag_point_selection_error == False:
                                    if (i_triangle_index > 0) and (i_triangle_point == side_choice_vertex[i_triangle_index-1]):
                                        if last_ii_index > point_selection_line_ii:
                                            polygon_vertex_re.insert(-1,(point_selection_line_x, point_selection_line_y))
                                        else:
                                            polygon_vertex_re.append((point_selection_line_x, point_selection_line_y))
                                    else:
                                        polygon_vertex_re.append((point_selection_line_x, point_selection_line_y))
                                    last_ii_index = copy.deepcopy(point_selection_line_ii)
                                else:
                                    # print("多边形障碍物生成失败，转为矩形障碍物 ", i_B_count)
                                    break
                                # print("polygon_vertex_re_2",polygon_vertex_re)
                            if flag_point_selection_error == False:
                                rect_b = polygon_vertex_re
                        flag_B_obstacle = True
                        area_B_rectang = Polygon(rect_b)
                        if area_B_rectang.is_valid == False:
                            print("\033[43;34m area_B_rectang的边界无效\033[0m")
                            print("\033[43;34m{}\033[0m".format(rect_b))
                            # print(hhhh_selection, flag_point_selection_error, polygon_vertex_re)
                            flag_B_obstacle = False
                        # print(rect_b)
                        # print("rect_b: ", rect_b, "typeL: ", obs_type_B) #rect_b出现过输出一条直线上的三个点的情况，导致intersection函数失败，没有深入找为什么会出现这种情况，只是检测了一下然后跳过了
                        for ipppp in range(len(rect_b)): #rect_b出现过输出的重复顶点的问题，这里直接跳过后期再调前面代码，并且和上面的问题都是三角形时出现的
                            if rect_b[ipppp] in rect_b[:ipppp]:
                               flag_B_obstacle = False
                               break
                        if flag_B_obstacle == True:
                            expansive_area_B_rectang = area_B_rectang.buffer(10,join_style=2,cap_style=2)
                            intersection_B_rectang = area_B_rectang.intersection(self.free_space_obstacles).area
                            intersection_expansive_B_rectang = expansive_area_B_rectang.intersection(self.free_space_obstacles).area
                            if abs(intersection_expansive_B_rectang - expansive_area_B_rectang.area) < 0.1:
                                if abs(intersection_B_rectang - area_B_rectang.area) < 0.1:
                                    contour_cross_count_b = 0
                                    for i_seg_points_b in range(len(points_test_segmentation)):
                                        p_lot_B = Point(points_test_segmentation[i_seg_points_b][0],points_test_segmentation[i_seg_points_b][1])
                                        if(p_lot_B.within(area_B_rectang)==True):
                                            contour_cross_count_b = contour_cross_count_b + 1
                                        if contour_cross_count_b > 0:
                                            flag_B_obstacle = False
                                            break
                                    if contour_cross_count_b > 0:
                                        # print("B障碍物存在与边界的交点1")
                                        flag_B_obstacle = False
                                else:
                                    flag_B_obstacle = False
                                    # print("B障碍物存在与边界的交点2")
                            else:
                                flag_B_obstacle = False
                                # print("B障碍物存在与边界的交点3")
                        # print("=== testing 8===") #2_11 test


                        # 判断条件：是否挡住门口
                        if flag_B_obstacle == True:
                            for doorway_polo in doorway_obastacle_testing:
                                if doorway_polo.intersection(self.free_space_obstacles).area  > 1:
                                    if area_B_rectang.intersection(doorway_polo).area > 1:
                                        flag_B_obstacle = False
                                        break
                            for doorway_polo in door_plank_testing:
                                # print("doora: ", doorway_polo.exterior() )
                                if doorway_polo.intersection(self.free_space_obstacles).area  > 1:
                                    if area_B_rectang.intersection(doorway_polo).area >= 1:
                                        flag_B_obstacle = False
                                        break

                        # 判断条件二：避免与已添加的障碍物靠太近
                        if flag_B_obstacle == True:
                            for try_obstacle_B in added_obstacles_list:
                                area_try_obstacle_B = Polygon(try_obstacle_B)
                                if expansive_area_B_rectang.intersection(area_try_obstacle_B).area > 0:
                                    flag_B_obstacle = False
                                    # print("B障碍物存在与边界障碍物的交点1")
                            
                            # B_dense_flag = 0
                            if B_dense_flag == 0:
                                for try_obstacle_B_2 in added_obstacles_list_B:
                                    area_try_obstacle_B_2 = Polygon(try_obstacle_B_2)
                                    if expansive_area_B_rectang.intersection(area_try_obstacle_B_2).area > 0:
                                        flag_B_obstacle = False
                                        # print("B障碍物存在与独立障碍物的交点2")
                        
                        # 判断条件三-停止条件：障碍物占据面积足够大
                        if flag_B_obstacle == True:
                            area = self.segmentation_area_properties[index_test]["new_area"]
                            if area < max((self.segmentation_area_properties[index_test]["origin_area"]*0.1), (1.5*self.config['meter2pixel']*self.config['meter2pixel'])):
                                flag_B_obstacle = False
                                # print("障碍物占据面积过大")
                                break



                        if flag_B_obstacle == True:
                        # 1/2处障碍物形状针对性处理-图形生成
                        
                            editable_map = np.zeros_like(self.world)
                            editable_map[self.world == self.map_color['obstacle']] = self.map_color['obstacle']
                            editable_map[self.world == self.map_color['free']] = self.map_color['free']
                            temp_obastacle_map = np.zeros_like(editable_map)
                            # print()
                            if (obs_type_B == 2) or (obs_type_B == 0):
                                draw_iscretization_no = False
                                if flag_2d_laser_lattice == 1:
                                    # print("rect_b: ", rect_b)
                                    draw_iscretization_no = self.draw_graphic_iscretization(obs_type_B, rect_b, obstacle_B_theta, temp_obastacle_map)
                                if flag_2d_laser_lattice == 0 or draw_iscretization_no == True:
                                    cv2.fillPoly(temp_obastacle_map, [np.array(rect_b,np.int32)], self.map_color["obstacle"])

                            else:
                                draw_iscretization_no = False
                                if flag_2d_laser_lattice == 1:
                                    draw_iscretization_no = self.draw_graphic_iscretization(obs_type_B, [int(round(obs_x_b)), int(round(obs_y_b)), int(round(obs_type_B_a_size)), int(round(obs_type_B_b_size)), rect_b], obstacle_B_theta, temp_obastacle_map)
                                if flag_2d_laser_lattice == 0 or draw_iscretization_no == True:
                                    cv2.ellipse(temp_obastacle_map, (obs_x_b,obs_y_b), (int(obs_type_B_a_size),int(obs_type_B_b_size)), obstacle_B_theta, 0,360, self.map_color["obstacle"],thickness=-1)

                            editable_map[temp_obastacle_map==0] = self.map_color['obstacle']
                            # temp_obastacle_map[editable_map==self.map_color['free']] = 145
                            # temp_obastacle_map[editable_map==self.map_color['obstacle']] = 145
                            # print("shape",temp_obastacle_map.shape, type(temp_obastacle_map),editable_map.shape, type(editable_map) )
                            self.world[editable_map==self.map_color['free']] = self.map_color['obstacle']
                            added_obstacles_list_B.append(copy.deepcopy(rect_b))
                            self.segmentation_area_properties[index_test]["new_area"] = self.segmentation_area_properties[index_test]["new_area"] - (area_B_rectang.area)
                            if flag_2d_laser_lattice == 0 or draw_iscretization_no == True:
                                temp_obastacle_map_3 = np.zeros_like(temp_obastacle_map,np.uint8)
                                temp_obastacle_map_3[temp_obastacle_map ==  self.map_color["obstacle"]] = 255
                                contours_A3, hierarchy_A1 = cv2.findContours(temp_obastacle_map_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
                                contours_A3 = contours_A3[0]
                                conto_ttt = []
                                for A3_p in contours_A3:
                                    A3_p = A3_p[0]
                                    conto_ttt.append([int(A3_p[0]), int(A3_p[1])])
                                if len(conto_ttt) > 4:
                                    if conto_ttt[-1] != conto_ttt[0]:
                                        conto_ttt.append(conto_ttt[0])
                                    B_contours.append(conto_ttt)
                            

                            # print("=== testing 9===") #2_11 test
                            # plt.clf()
                            # plt.imshow(self.world,cmap="gray")
                            # plt.draw()
                            # plt.pause(0.01)
                            # dic_tmp = {0:"矩形",1:"椭圆",2:"多边形"}
                            # print("生成障碍物形状: ",dic_tmp[obs_type_B])
                            # print("障碍物中心位置: ",(obs_x_b,obs_y_b))
                            # print("多边形基准顶点位置: ",rect_b)
                            # print("障碍物长宽(半): ", obs_type_B_a_size, obs_type_B_b_size)
                            # print("生成后的障碍物长宽: ", B_side_1_test, B_side_2_test,"误差: ",(max(B_side_1_test,B_side_2_test) - obs_type_B_a_size*2),(min(B_side_1_test,B_side_2_test) - obs_type_B_b_size*2))
                            # print("障碍物旋转角度: ", obstacle_B_theta)
                            # print("独立障碍物是否重合: ",{0:"不重合", 1:"重合"}[B_dense_flag])
                            # print("独立障碍物是否对齐房间方向: ",{0:"对齐", 1:"不对齐"}[B_direction_flag],"是否检测到主方向: ",len(main_directions))
                            flag_b_ok = True
                        i_B_count = i_B_count + 1
                    searching_times_1_b = searching_times_1_b + 1
            if len(boundary_append_obstacle_points) > 0:
                tmp_shring = []
                last_dic_sign = 1
                for tmp_shring_index in boundary_append_obstacle_points_list:
                    if tmp_shring_index >=0:
                        tmp_shring.append(points_test_segmentation[tmp_shring_index])
                    else:
                        if last_dic_sign >= 0:
                            for i_mm_ in boundary_append_obstacle_points[tmp_shring_index]:
                                tmp_shring.append(i_mm_)
                        elif last_dic_sign !=  tmp_shring_index:
                            for i_mm_ in boundary_append_obstacle_points[tmp_shring_index]:
                                tmp_shring.append(i_mm_)
                    last_dic_sign = tmp_shring_index
                self.segmentation_points_dynamic_shrinking[index_test]=tmp_shring
                # print("pt2: ", tmp_shring)
            # self.add_
            All_B_contours.append(B_contours)  

            end_time2 = time.time()
            elapsed_time3 = end_time2 - building_start_time
            elapsed_time4 = end_time2 - building_ll_time
            print("Time-IN-B: {:.2f}-{:.2f}".format(elapsed_time4, elapsed_time3))
            building_ll_time = time.time()
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            elapsed_time2 = end_time - self.ll_time
            print("房间模拟成功 Time-room {:.2f}s-{:.2f}s".format(elapsed_time2, elapsed_time))
            self.ll_time = time.time()

        if self.dispaly_flag == True:
            if self.data_generation_RL_flag == False:
                self.show_contour_pic(self.world, self.segmentation_points_dynamic_shrinking, 'complete room',stop=False)
                self.show_contour_pic(self.world, self.segmentation_points, 'complete room',stop=False)
                self.show_contour_pic(self.world, [], 'origin room',stop=False)

        self.noise_world = np.zeros_like(self.world)
        self.noise_world[self.world == self.map_color['obstacle']] = self.map_color['obstacle']
        self.noise_world[self.world == self.map_color['free']] = self.map_color['free']
        for index_test, points_test_segmentation in enumerate(self.segmentation_points):
            free_space_obstacles_2 = Polygon(points_test_segmentation)
            bbox = free_space_obstacles_2.bounds
            min_x,min_y,max_x,max_y = bbox[0],bbox[1],bbox[2],bbox[3]
            noise1, noise2, noise3 = self.map_slam_laser_jont_error(self.noise_world, self.segmentation_points_dynamic_shrinking[index_test], min_x,min_y,max_x,max_y, all_point_lasers_home, All_B_contours[index_test])
            noise_info_.append([noise3])
        self.noise_world_flag = True

        # self.show_contour_pic(self.noise_world, self.segmentation_points_dynamic_shrinking, 'complete room',stop=False)
        # self.show_contour_pic(self.noise_world, self.segmentation_points, 'complete room',stop=False)

        # self.show_contour_pic(self.noise_world, [], 'noise room',stop=False)

        # print("noise_info_: ",noise_info_)
        # print("=== testing 10===") #2_11 test

        # approx_segmentation = []
        # for change_index, change_segmentation in enumerate(self.segmentation_points):
        #     convex_temp = np.zeros_like(self.world,np.uint8)
        #     convex_temp_2 = np.zeros_like(self.world,np.uint8)
        #     # origin_s = Polygon(change_segmentation)
        #     shrink_s = Polygon(self.segmentation_points_dynamic_shrinking[change_index])
        #     shrink_convex = shrink_s.convex_hull
        #     bbox = shrink_convex.bounds
        #     min_x_shrink,min_y_shrink,max_x_shrink,max_y_shrink = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        #     for i in range(min_x_shrink, max_x_shrink + 1):
        #         for j in range(min_y_shrink, max_y_shrink + 1):
        #             p_lot = Point(i,j)
        #             if(p_lot.within(shrink_convex)==True or p_lot.within(shrink_convex.boundary)==True):
        #                 convex_temp[j][i] = 255
        #     cv2.fillPoly(convex_temp_2, [np.array(change_segmentation,np.int32)],255)
        #     convex_temp[convex_temp_2 == 0] = 0
        #     contours_change, hierarchy_change = cv2.findContours(convex_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
        #     contours_change = contours_change[0]
        #     change_ttt = []
        #     for A3_p in contours_change:
        #         A3_p = A3_p[0]
        #         if self.world[int(A3_p[1])][int(A3_p[0])] != self.map_color['free']:
        #             change_ttt.append([int(A3_p[0]), int(A3_p[1])])
        #     #考虑这里是否还加收尾重复，findContours结果是没有收尾重复的
        #     approx_segmentation.append(change_ttt)
        #     # print("contours_change all: ", len(contours_change), "contours_change_new", len(change_ttt))
        # if self.dispaly_flag == True:
        #     if self.data_generation_RL_flag == False:
        #         self.show_contour_pic(self.world, approx_segmentation, 'approx room',stop=False)


        end_time = time.time()
        elapsed_time = end_time - self.start_time
        elapsed_time2 = end_time - self.ll_time
        print("Time-Homes {:.2f}s-{:.2f}s\n".format(elapsed_time2, elapsed_time))
        self.ll_time = time.time()
        return True
            # print("$$$C类型障碍物生成$$$")
            # input("等待继续...")
            # obstacle_C_num = 5
            # point_num_obstacles_c = 0
            # for _______ in range(obstacle_C_num):
            #     searching_times_1_c = 0
            #     point_num_obstacles_c = point_num_obstacles_c + 1
            #     print("\n$$ {}/{} $$".format(point_num_obstacles_c,obstacle_C_num))
            #     # input("等待继续...")
            #     obs_type_C = np.random.choice([0,1,2],p=[0.35,0.3,0.35])
            #     if sensor_type == 0:
            #         flag_2d_laser_lattice = np.random.choice([0,1],p=[0.5,0.5])
            #     else:
            #         flag_2d_laser_lattice = 0
            #     is_obstacles_close_C = np.random.choice([0,1],p=[0.8,0.2])
            #     if is_obstacles_close_C == 0:
            #         print("此障碍物不接触其他障碍物")
            #     else:
            #         print("此障碍物可能接触其他障碍物")
            #     print("---整体形状(0矩形;1椭圆形;2三角形):",obs_type_C)
            #     flag_c_ok = False
            #     while (searching_times_1_c < 15) and (flag_c_ok == False):
            #         print("      {}/15".format(searching_times_1_c))
            #         obs_type_C_a_size, obs_type_C_b_size= self.random_multi_obstacle_size(0, obs_type_C, obs_sizeRange_room)
            #         area = self.segmentation_area_properties[index_test]["new_area"]
            #         area_limit = min((area*0.7), area-(1.5*self.config['meter2pixel']*self.config['meter2pixel']))
            #         obs_type_C_area = obs_type_C_a_size * obs_type_C_b_size
            #         cout_area_limit_searching_c = 1
            #         go_next_flag_c = True
            #         while (obs_type_C_area > area_limit) and cout_area_limit_searching_c < 200:
            #             print("**所确定的障碍物尺寸的面积校验失败---重新确定尺寸, 旧尺寸:{} **".format(obs_sizeRange_room))
            #             if cout_area_limit_searching_c % 50 == 0:
            #                 obs_sizeRange_room_ttt = []
            #                 length_limit = length_limit * 0.8
            #                 for seg_range in obs_sizeRange_room:
            #                     if length_limit < seg_range[1]:
            #                         if length_limit <= seg_range[0]:
            #                             pass
            #                         else:
            #                             obs_sizeRange_room_ttt.append([seg_range[0], area_limit])
            #                     else:
            #                         obs_sizeRange_room_ttt.append(seg_range)
                            
            #                 obs_sizeRange_room = obs_sizeRange_room_ttt
            #             obs_type_C_a_size, obs_type_C_b_size= self.random_multi_obstacle_size(0, obs_type_C, obs_sizeRange_room)
            #             obs_type_C_area = obs_type_C_a_size * obs_type_C_b_size

            #             print("新尺寸: ", obs_sizeRange_room)
            #             cout_area_limit_searching_c = cout_area_limit_searching_c + 1
            #             if len(obs_sizeRange_room) == 0:
            #                 go_next_flag_c = False
            #         if cout_area_limit_searching_c == 200:
            #             go_next_flag_c = False
            #         if go_next_flag_c == False:
            #             break
            #         obs_type_C_a_size = obs_type_C_a_size/2
            #         obs_type_C_b_size = obs_type_C_b_size/2
            #         i_C_count = 0
    def add_obstacle(self):
        """ Randomly add obstacle to world """
        if self.obstacle_config:
            """ add user defined obstacles """
            f = open(self.obstacle_config, "r")
            for config in f:
                x, y, w, h = [int(n) for n in config.split(" ")]
                rect = np.array([[x - w // 2, y - h // 2],
                                 [x - w // 2, y + h // 2],
                                 [x + w // 2, y + h // 2],
                                 [x + w // 2, y - h // 2]],
                                np.int32)
                cv2.fillPoly(self.world, [rect], self.map_color["obstacle"])
            return self.world

        if self.obs_num == 0:  # No obstacle added.
            return

        

        prox_min = 3 # min distance in pixel between added obstacle & obstacle in map
        world_obs = np.copy(self.world) # create a new image for adding obstacle
        (h,w)= self.world.shape

        for i in range(self.obs_num):
            JJ = 0
            while (1):
                JJ += 1
                if JJ > 5000:
                    break
                # randomly generate obstacle orientation & obstacle size that fall within obstacle_sizeRange
                obs_a = np.random.randint(self.obs_sizeRange[0], self.obs_sizeRange[1])*0.5
                obs_b = np.random.randint(self.obs_sizeRange[0], self.obs_sizeRange[1])*0.5
                obs_theta = np.random.random()*360

                # check if obstacle length exceed 45% of world shape, if yes, randomly change obs_sizeRange to 5%~30% of min(h,w)
                obs_len= np.sqrt(obs_a*obs_a + obs_b*obs_b)*2
                if (obs_len*2>h*0.9 or obs_len*2>w*0.9):
                    self.obs_sizeRange[0]= np.random.randint(500,1500)*1.0/10000 *np.min((h,w))
                    self.obs_sizeRange[1]= np.random.randint(500,3000)*1.0/10000 *np.min((h,w))
                    self.obs_sizeRange= np.sort(self.obs_sizeRange,axis=0).astype(int)
                    print('obs_sizeRange exceed world shape, now changing to ',self.obs_sizeRange.tolist())
                    continue

                # randomly select shape type of obstacle [0: rectangle; 1: ellipse; 2: circle]
                obs_type = np.random.randint(0, 3)
                if obs_type == 2:
                    obs_b = obs_a
                
                doorway = self.search_doorway()

                # randomly generate obstacle center coordinate that the obstacle would not exceed world boundary
                bound = np.round(obs_len).astype(int) + prox_min
                obs_y = np.random.randint(bound, h-bound)
                obs_x = np.random.randint(bound, w-bound)


                # check if the location of obstacle to be added intersect with other obstacle present
                # bound,obs_y, obs_x are arrays, should convert to inter
                # print('self.obs_sizeRange[0], self.obs_sizeRange[1]',self.obs_sizeRange[0], self.obs_sizeRange[1])
                
                bound = bound[0]
                obs_y= obs_y[0]
                obs_x = obs_x[0]
                obs_a = obs_a[0]
                obs_b = obs_b[0]
                if np.sum(world_obs[obs_y-bound:obs_y+bound+1, obs_x-bound:obs_x+bound+1]==self.map_color['obstacle'])!= 0:
                    continue

                
                # create obstacle patch
                if obs_type == 0:
                    cthe = np.cos(np.pi/180* obs_theta)
                    sthe = np.sin(np.pi/180* obs_theta)
                    rect = np.array([[obs_x + (-obs_a * cthe - -obs_b * sthe), obs_y + (-obs_a * sthe + -obs_b * cthe)],
                                     [obs_x + (-obs_a * cthe - obs_b * sthe), obs_y + (-obs_a * sthe + obs_b * cthe)],
                                     [obs_x + (obs_a * cthe - obs_b * sthe), obs_y + (obs_a * sthe + obs_b * cthe)],
                                     [obs_x + (obs_a * cthe - -obs_b * sthe), obs_y + (obs_a * sthe + -obs_b * cthe)]],
                                    np.int32)
                    # print('rect,cthe,sthe,obs_a,obs_b:',rect,cthe,sthe,obs_a,obs_b)
                    cv2.fillPoly(world_obs, [rect], self.map_color["obstacle"])

                elif obs_type == 1:
                    cv2.ellipse(world_obs, (obs_x,obs_y), (int(obs_a),int(obs_b)), obs_theta, 0,360, self.map_color["obstacle"],thickness=-1)
                else:
                    cv2.circle(world_obs, (obs_x,obs_y), int(obs_a), self.map_color["obstacle"],thickness=-1)

                break

        self.world= world_obs.copy()

        return world_obs

    def _randomizeRobotPose(self):
        # randomly generate robot start pose where robot is not crashed into obstacle
        h, w = self.world.shape
        x_min, x_max = int(0.1 * w), int(0.8 * w)
        y_min, y_max = int(0.1 * h), int(0.8 * h)
        self.robotPose[0] = np.random.randint(y_min, y_max)
        self.robotPose[1] = np.random.randint(x_min, x_max)

        while (self.robotCrashed(self.robotPose)):
            self.robotPose[0] = np.random.randint(y_min, y_max)
            self.robotPose[1] = np.random.randint(x_min, x_max)
        self.robotPose[2] = np.random.rand() * np.pi * 2
        return self.robotPose


    def _randomizeRobotPose_room(self):
        if self.passable_points == []:
            print('\033[1;31;47m RandomizeRobotPose failed!!!\033[0m')
            # self._randomizeRobotPose()
        else:
            randomize_pose_order = np.random.randint(0,len(self.passable_points))
            self.robotPose[0] = self.passable_points[randomize_pose_order][0]
            self.robotPose[1] = self.passable_points[randomize_pose_order][1]
            self.robotPose[2] = np.random.rand() * np.pi * 2
            while (self.robotCrashed(self.robotPose)):
                randomize_pose_order = np.random.randint(0,len(self.passable_points))
                self.robotPose[0] = self.passable_points[randomize_pose_order][0]
                self.robotPose[1] = self.passable_points[randomize_pose_order][1]
                self.robotPose[2] = np.random.rand() * np.pi * 2
            print("RobotPose_room")
        return self.robotPose


    def reset(self, order=True):
        reset_start_time = time.time()
        reset_ll_time = time.time()
        print("Now Order: ", self.order_id_pointer)
        self.traj.clear()
        self.scaning_noise_type = np.random.choice([0,1],p=[0.2,0.8])
        if self.noise_world_flag == False:
            while (not self.create_world(order)) and (not self.traverse_end):
                self.skip_map_id.append(self.map_id)
                print('\033[1;31;47m Skip a Map - {}\033[0m'.format(self.map_id))
                print("\033[31m Skipped already: {}\033[0m".format(self.skip_map_id))
            reset_end_time = time.time()
            reset_elapsed_time = reset_end_time - reset_start_time
            reset_elapsed_time2 = reset_end_time - reset_ll_time
            print("Household Obstacle completed Time: {:.2f}s-{:.2f}s".format(reset_elapsed_time, reset_elapsed_time2))
            reset_ll_time = time.time()
        else:
            self.world = self.noise_world
            self.noise_world_flag = False
        if self.dispaly_flag == True:
            if self.data_generation_RL_flag == False:
                self.show_contour_pic(self.world, self.segmentation_points_dynamic_shrinking, 'complete room',stop=False)
                self.show_contour_pic(self.world, self.segmentation_points, 'complete room',stop=False)
                self.show_contour_pic(self.world, [], 'origin room',stop=False)
            else:
                self.show_contour_pic(self.world, [], 'origin room',stop=False)
        self.passable_points = []
        if self.complete_construciton == True:
            self.slamMap= np.ones(self.world.shape)*self.map_color["uncertain"]
            # print("time_test_1")
            ______, pic_com = self.connect_area_slam()
            # print("time_test_2")
            self.Generation_complete_map(pic_com)
            self.comMap = self.slamMap
            reset_end_time = time.time()
            reset_elapsed_time = reset_end_time - reset_start_time
            reset_elapsed_time2 = reset_end_time - reset_ll_time
            print("comMap constructed: {:.2f}s-{:.2f}s".format(reset_elapsed_time, reset_elapsed_time2))
            reset_ll_time = time.time()

        if (self.robotResetRandomPose==1) or (self.robotCrashed(self.robotPose_init)):
            # randomly generate robot start pose where robot is not crashed into obstacle
            self._randomizeRobotPose_room()
        else:
            self.robotPose = self.robotPose_init
        reset_end_time = time.time()
        reset_elapsed_time = reset_end_time - reset_start_time
        reset_elapsed_time2 = reset_end_time - reset_ll_time
        print("Robot Pose completed: {:.2f}s-{:.2f}s".format(reset_elapsed_time, reset_elapsed_time2))
        reset_ll_time = time.time()
        self.robotCrashed_flag= False
        if self.is_exploration:
            self.slamMap= np.ones(self.world.shape)*self.map_color["uncertain"]
            self.dslamMap= np.ones(self.world.shape)*self.map_color["uncertain"]
        else:
            self.slamMap = self.world.copy()
            self.dslamMap = self.world.copy()
        self.build_map(self.robotPose)
        return


    def _laser_noise(self, y_rangeCoordMat, x_rangeCoordMat, y_coord, x_coord, b):
        """ add laser noise, y_&x_coord are coord before obstacle | y_&x_rangeCoordMat are coord that the laser range covers
        b is the vector that represent the index of rangeCoordMat where the coord are before obstacle """

        noise_vector = np.random.normal(loc=0, scale=self.laser_noiseSigma, size=(x_rangeCoordMat.shape[0], 1))
        noise_vector = np.round(noise_vector).astype(np.int64)
        #noise_vector的shape是(180,1),列向量。
        # 将noise_vector加到y_rangeCoordMat和x_rangeCoordMat中，也就是说每条扫描射线上的点的坐标都进行相同的噪声偏移，并且x和y偏移是一样的，是个正方形偏移
        # 但不同的扫描射线上的点的坐标偏移应该是不同的

        # print("_laser_noise: ", noise_vector.shape,noise_vector)
        """ add noise to rangeCoordMat  """
        y_rangeCoordMat += noise_vector
        x_rangeCoordMat += noise_vector
        y_noise_coord = y_rangeCoordMat[b]
        x_noise_coord = x_rangeCoordMat[b]
        #b是根据之前射线上障碍物的分布得到的距离最近的障碍物边界分布，一旦加上noise_vector，y_rangeCoordMat中障碍物分布就不一样了
        #比如根据原来的原来y_rangeCoordMat分布，b是取到是直线上距离中心点某个距离以内所有点
        # y_rangeCoordMat加上噪声后去的点的分布变了，可能y_rangeCoordMat的行不再是射线上的点，以直线为中心出现了坐标偏移，因此，利用b得到的点集可能不完全是空闲区域和障碍物最外侧点，是否越界也需要重新计算了
        """ check for index of the coord that are within bound of world """
        inBound_ind= util.within_bound(np.array([y_noise_coord, x_noise_coord]),self.world.shape)
        #上面实际上是简化版的""" delete coordinate that are not within the bound """，采用的删除越界点，而非像之前一样的替换，可能因为这里不要后续计算了
        """ get the coord that are within bound """
        x_noise_coord = x_noise_coord[inBound_ind]
        y_noise_coord = y_noise_coord[inBound_ind]
        x_coord = x_coord[inBound_ind]
        y_coord = y_coord[inBound_ind]
        #这里之所以对x_coord也进行inBound_ind限制，只是为了让噪声前后矩阵保持索引一致，并不是说x_coord存在越界点

        return y_noise_coord, x_noise_coord, y_coord, x_coord
    
    def _laser_noise_for_value(self, y_rangeCoordMat, x_rangeCoordMat, obstacle_ind):
        """ add laser noise, y_&x_coord are coord before obstacle | y_&x_rangeCoordMat are coord that the laser range covers
        b is the vector that represent the index of rangeCoordMat where the coord are before obstacle """
        #这里应该不需要计算点是否超出地图有效范围，因为y_rangeCoordMat中超出地图范围的索引已经被替换为外最外侧的有效点。
        #我只是对b（索引向量）增加了噪声，没有对y_rangeCoordMat本身的值添加噪声
        # print("shape of obstacle_ind in function: ", obstacle_ind.shape)
        obstacle_ind_noise = np.zeros_like(obstacle_ind)
        noise_vector1 = np.random.normal(loc=0, scale=self.laser_noiseSigma, size=len(obstacle_ind_noise))
        # if noise_vector1 > 0:
        #     noise_vector1 = -noise_vector1
        # if type == 0:
        #     noise_vector2 = np.zeros(len(obstacle_ind_noise))
        # if type == 1:
        #     noise_vector2 = np.random.normal(loc=1, scale=(self.laser_range + 1)/4.0, size=len(obstacle_ind_noise))
        # if type == 2:
        #     noise_vector2=np.ones(len(obstacle_ind_noise))
        #     noise_vector2[:] = np.random.normal(loc=1, scale=(self.laser_range + 1)/4.0)
        noise_vector1 = np.round(noise_vector1).astype(np.int64)
        # noise_vector2 = np.round(noise_vector2).astype(np.int64)
        for ii_ind in range(len(obstacle_ind)):
            if obstacle_ind[ii_ind] == x_rangeCoordMat.shape[1]:
                obstacle_ind_noise[ii_ind] = obstacle_ind[ii_ind]
            else:
                min_ind = 1- obstacle_ind[ii_ind]
                max_ind = x_rangeCoordMat.shape[1] - obstacle_ind[ii_ind]
                # noise_a = noise_vector1[ii_ind] + noise_vector2[ii_ind]
                noise_a = noise_vector1[ii_ind]
                if noise_a > 0:
                    noise_a = -noise_a
                # gross_noise = np.random.normal(loc=1, scale=max_ind/2.0, size=len(obstacle_ind_noise))
                if noise_a < min_ind:
                    noise_a = min_ind
                if noise_a > max_ind:
                    noise_a = max_ind
                obstacle_ind_noise[ii_ind] = obstacle_ind[ii_ind] + noise_a
        bx_new_free = np.arange(x_rangeCoordMat.shape[1]).reshape(1, x_rangeCoordMat.shape[1])
        by_new_free = np.ones((x_rangeCoordMat.shape[0], 1))
        b_new_free = np.matmul(by_new_free, bx_new_free)
        b_new_free = b_new_free < obstacle_ind_noise.reshape(obstacle_ind_noise.shape[0], 1)
        y_noise_coord_free = y_rangeCoordMat[b_new_free]
        x_noise_coord_free = x_rangeCoordMat[b_new_free]
        bx_new_obs = np.arange(x_rangeCoordMat.shape[1]).reshape(1, x_rangeCoordMat.shape[1])
        by_new_obs = np.ones((x_rangeCoordMat.shape[0], 1))
        b_new_obs = np.matmul(by_new_obs, bx_new_obs)
        b_new_obs = b_new_obs == obstacle_ind_noise.reshape(obstacle_ind_noise.shape[0], 1)
        y_noise_coord_obs = y_rangeCoordMat[b_new_obs]
        x_noise_coord_obs = x_rangeCoordMat[b_new_obs]
        return y_noise_coord_free, x_noise_coord_free, y_noise_coord_obs, x_noise_coord_obs





    def _slam_error(self, y_coord, x_coord):
        #_slam_error与_laser_noise过程不同，结果差不多。
        # _laser_noise是依照扫描射线设置误差，使得射线上的点扫偏了，扫到了不正确的位置。
        #而_slam_error则是对整个扫描区域y_noise_rangeCoordMat, x_noise_rangeCoordMat进行相同的旋转与偏移。
        # 旋转偏移的具体过程是按照机器人当前的朝向（即依据局部坐标系）对环境结构中的点（也就是扫描点）进行绕机器人旋转与平移。
        #_laser_noise是根据激光扫描射线制造误差，_slam_error是整体扫描区域制造误差，结果都是将环境结构中的点（self.world）映射到了slam地图中的错误位置
        err_y = util.gauss_noise() * self.slamErr_linear
        err_x = util.gauss_noise() * self.slamErr_linear
        err_theta = util.gauss_noise() * self.slamErr_angular

        """ rotate y_coord & x_coord by err_theta """
        (y_coord_err, x_coord_err)= util.transform_coord(y_coord,x_coord, self.robotPose, np.array([err_y,err_x,err_theta]))
        #shape of np.array([err_y,err_x,err_theta]):(3,)是个三个数的行向量，也就是对所有的y_coord,x_coord点施加相同的噪声
        """ check for index where the coord are within bound """
        inBound_ind= util.within_bound(np.array([y_coord_err,x_coord_err]),self.world.shape)
        x_err_coord= x_coord_err[inBound_ind]
        y_err_coord= y_coord_err[inBound_ind]

        x_coord = x_coord.reshape(x_coord.shape[0], 1)[inBound_ind]
        y_coord = y_coord.reshape(y_coord.shape[0], 1)[inBound_ind]

        return y_err_coord, x_err_coord, y_coord, x_coord

    def map_slam_laser_jont_error(self, origin_world, _room_boundary_,min_x_l,min_y_l,max_x_l,max_y_l, all_point_lasers, B_type_boundary):
        """laser_noise 1"""
        laser_noise_turn_on = np.random.choice([0,1],p=[0.15,0.85])
        def if_black_conotur_pixel_100(image,raw, i):
            if image[raw][i] == self.map_color['obstacle']:
                t = image[max(raw-1,0):min(raw+2, len(image)),max(i-1,0):min(i+2,len(image[raw]))]
                if (t == self.map_color['free']).any():
                    return True
            return False
        laser_noise_num = np.random.choice([1,2,3])


        # laser_noise_turn_on = 0 #test



        if laser_noise_turn_on == 0:
            laser_noise_num = 0
        for ___ in range(laser_noise_num):
            obs_y = int(np.random.randint(min_y_l,max_y_l))
            obs_x = int(np.random.randint(min_x_l,max_x_l))
            laser_center = Point(obs_x, obs_y)
            # print("len of _room_boundary_: ", len(_room_boundary_))
            area_room = Polygon(_room_boundary_)
            while origin_world[obs_y][obs_x] == self.map_color["obstacle"] or laser_center.within(area_room)==False:
                obs_y = int(np.random.randint(min_y_l,max_y_l))
                obs_x = int(np.random.randint(min_x_l,max_x_l))
                laser_center = Point(obs_x, obs_y)
            obs_theta = np.random.random() * math.pi * 2
            lasers_info = 0
            stop_times = 0
            lasers = np.random.choice([1,2,3,4,5])
            while lasers_info < lasers and stop_times < 1000:
                if stop_times == 0:
                    ttheta = obs_theta
                    # lasers_info.append(obs_theta)
                else:
                    ttheta = obs_theta + ((np.random.random()-0.5) * 2 * math.pi /4)
                    # lasers_info.append(obs_theta + ((np.random.random()-0.5) * 2 * math.pi /4))
                length_laser = int(np.random.random()*self.laser_range/2.0 + self.laser_range/2.0)
                len_seg = [i for i in range(length_laser+1)]
                num_cross = 0
                num_black = 0
                _laser_temp = []
                for point in len_seg:
                    y_ppp = int(round(obs_y - point * math.sin(ttheta)))
                    x_ppp = int(round(obs_x + point * math.cos(ttheta)))
                    if (y_ppp >= 0 and y_ppp < origin_world.shape[0] ) and (x_ppp>=0 and x_ppp < origin_world.shape[1]):
                        if [x_ppp, y_ppp] not in _laser_temp:
                            if [x_ppp, y_ppp] in all_point_lasers:
                                num_cross = num_cross + 1
                            if origin_world[y_ppp][x_ppp] == self.map_color["obstacle"]:
                                num_black = num_black + 1
                            _laser_temp.append([x_ppp, y_ppp])
                if num_cross < 3 and num_black > 0:
                    all_point_lasers = all_point_lasers + _laser_temp
                    for change_point in _laser_temp:
                        origin_world[change_point[1]][change_point[0]] = self.map_color["free"]
                    # print("laser_noise 1 #", lasers_info,"/",___,"stop_times",stop_times)
                    lasers_info = lasers_info + 1
                stop_times = stop_times + 1
        # print("1-20")
        # self.show_contour_pic(origin_world, [], 'noise origin room-0',stop=False)
        """laser_noise 2"""
        laser_noise_2_turn_on = np.random.choice([0,1],p=[0.15,0.85])
        free_grid_noise_num = np.random.choice([1,2,3,4,5,6,7,8,9,10])


        # laser_noise_2_turn_on = 0 #test

        if laser_noise_2_turn_on == 0:
            free_grid_noise_num = 0
        lasers_info2 = 0
        stop_times = 0
        while lasers_info2 < free_grid_noise_num and stop_times < 1000:
            obs_y = int(np.random.randint(min_y_l,max_y_l))
            obs_x = int(np.random.randint(min_x_l,max_x_l))
            laser_center = Point(obs_x, obs_y)
            while origin_world[obs_y][obs_x] == self.map_color["obstacle"] or laser_center.within(Polygon(_room_boundary_))==False:
                obs_y = int(np.random.randint(min_y_l,max_y_l))
                obs_x = int(np.random.randint(min_x_l,max_x_l))
                laser_center = Point(obs_x, obs_y)
            if (obs_y >= 0 and obs_y < origin_world.shape[0]) and (obs_x>=0 and obs_x < origin_world.shape[1]):
                if [obs_x, obs_y] not in all_point_lasers:
                    if origin_world[obs_y][obs_x] == self.map_color["free"]:
                        origin_world[obs_y][obs_x] = self.map_color["obstacle"]
                    lasers_info2 = lasers_info2 + 1
                    # print("laser_noise 2 #", lasers_info2,"/",free_grid_noise_num,"stop_times",stop_times)
            stop_times = stop_times + 1
        # print("1-21")
        """slam_noise"""
        laser_slam_turn_on = np.random.choice([0,1],p=[0.15,0.85])


        stop_times = 0
        slam_noise_num = np.random.choice([1,2,3,4,5,6])

        # laser_slam_turn_on = 0 #test


        if laser_slam_turn_on == 0:
            slam_noise_num = 0
        slam_info = 0
        point_num = len(_room_boundary_) - 1
        curve_min = min(5, point_num/10.0)
        cruve_max = point_num/2.0
        if point_num < 5:
            slam_noise_num = 0
        while slam_info < slam_noise_num and stop_times < 1000:
            boundary_type = np.random.choice([0,1],p=[0.7,0.3])


 
            if len(B_type_boundary) == 0:
                boundary_type = 0
            if boundary_type == 0:
                # print("-----")
                # print("boundary")
                # print("outer obs slam noies")
                if curve_min >= cruve_max:
                    boundary_width = curve_min
                else:
                    boundary_width = curve_min + (cruve_max - curve_min) * np.random.random()
                boundary_width = int(round(boundary_width))
                source_curve = []
                start_point_index = int(round(np.random.random() * point_num))
                if (start_point_index + boundary_width) <= (len(_room_boundary_) - 2):
                    for point_index_ in range(start_point_index, start_point_index + boundary_width):
                        source_curve.append(_room_boundary_[point_index_])
                else:
                    for point_index_ in range(start_point_index, len(_room_boundary_)-1):
                        source_curve.append(_room_boundary_[point_index_])
                    for point_index_ in range(boundary_width-(len(_room_boundary_) - 1 - start_point_index)):
                        source_curve.append(_room_boundary_[point_index_])
            
            else:
                # print("inner obs slam noies")
                # print("disperse")
                b_obs_order = np.random.choice(list(range(len(B_type_boundary))))
                b_min = 5
                b_max = len(B_type_boundary[b_obs_order])/2
                if b_min >= b_max:
                    boundary_width = b_min
                else:
                    boundary_width = int(round(b_min+(b_max - b_min) * np.random.random()))
                source_curve = []
                start_point_index = int(round(np.random.random() * (len(B_type_boundary[b_obs_order]) - 1)))
                if (start_point_index + boundary_width) <= (len(B_type_boundary[b_obs_order]) - 2):
                    for point_index_ in range(start_point_index, start_point_index + boundary_width):
                        source_curve.append(B_type_boundary[b_obs_order][point_index_])
                else:
                    for point_index_ in range(start_point_index, len(B_type_boundary[b_obs_order])-1):
                        source_curve.append(B_type_boundary[b_obs_order][point_index_])
                    for point_index_ in range(boundary_width-(len(B_type_boundary[b_obs_order]) - 1 - start_point_index)):
                        source_curve.append(B_type_boundary[b_obs_order][point_index_])
                
            
            curve_dilatation = []
            for i in range(len(source_curve)):
                if i < len(source_curve) - 1:
                    if [int(source_curve[i][0]), int(source_curve[i][1])] not in curve_dilatation:
                        curve_dilatation.append([int(source_curve[i][0]), int(source_curve[i][1])])
                    theta_w = math.atan2(source_curve[i+1][1] - source_curve[i][1], source_curve[i+1][0] - source_curve[i][0])
                    curve_len = math.sqrt((source_curve[i][1] - source_curve[i+1][1])*(source_curve[i][1] - source_curve[i+1][1]) + (source_curve[i][0] - source_curve[i+1][0])*(source_curve[i][0] - source_curve[i+1][0]))
                    gap = 0.3
                    grp_num=[]
                    end_gap = 0.3
                    while end_gap < curve_len:
                        grp_num.append(end_gap)
                        end_gap = end_gap + gap
                    for _i_ in grp_num:
                        y_a = int(round(source_curve[i][1] + _i_ * math.sin(theta_w)))
                        x_a = int(round(source_curve[i][0] + _i_ * math.cos(theta_w)))
                        if [x_a, y_a] not in curve_dilatation:
                            if x_a < self.world.shape[1] and y_a < self.world.shape[0]:
                                if x_a >= 0 and y_a >= 0:
                                    curve_dilatation.append([x_a, y_a])   
                else:
                    if [int(source_curve[i][0]), int(source_curve[i][1])] not in curve_dilatation:
                        curve_dilatation.append([int(source_curve[i][0]), int(source_curve[i][1])])
            # print("boundary_width: ", boundary_width, "({})".format(point_num), "  source_curve: ", len(source_curve))
            # print("-> curve_dilatation: ", len(curve_dilatation))
            source_curve = []       
            slam_error_dilatation = np.random.choice([0,1],p=[0.7,0.3])



            for dense_point in curve_dilatation:
                ok_sig = False
                if slam_error_dilatation == 0:
                    # print("thin")
                    if if_black_conotur_pixel_100(self.world, dense_point[1], dense_point[0]):
                        source_curve.append(dense_point)
                    else:
                        break_sig = False
                        for i_1 in [-1,1,-2,2]:
                            for i_0 in [-1,1,-2,2]:
                                if dense_point[0] + i_0 >= 0 and dense_point[0] + i_0 < origin_world.shape[1]:
                                    if dense_point[1] + i_1 >= 0 and dense_point[1] + i_1 < origin_world.shape[0]:
                                        if if_black_conotur_pixel_100(self.world, dense_point[1] + i_1, dense_point[0] + i_0):
                                            source_curve.append([dense_point[0] + i_0, dense_point[1] + i_1])
                                            break_sig = True
                                            break
                            if break_sig == True:
                                break
            
                    # if self.world[dense_point[1]][dense_point[0]] == self.map_color["obstacle"]:
                    #     source_curve.append(dense_point)
                    # else:
                    #     break_sig = False
                    #     for i_1 in range(-1,2):
                    #         for i_0 in range(-1,2):
                    #             if dense_point[0] + i_0 >= 0 and dense_point[0] + i_0 < origin_world.shape[1]:
                    #                 if dense_point[1] + i_1 >= 0 and dense_point[1] + i_1 < origin_world.shape[0]:
                    #                     if self.world[dense_point[1] + i_1][dense_point[0] + i_0] == self.map_color["obstacle"]:
                    #                         source_curve.append([dense_point[0] + i_0, dense_point[1] + i_1])
                    #                         break_sig = True
                    #                         break
                    #         if break_sig == True:
                    #             break
                else:
                    range_i = np.random.choice([1,2,3])
                    # print("wide ({})".format(range_i))
                    for i_1 in range(-range_i,range_i+1):
                        for i_0 in range(-range_i,range_i+1):
                            if dense_point[0] + i_0 >= 0 and dense_point[0] + i_0 < origin_world.shape[1]:
                                if dense_point[1] + i_1 >= 0 and dense_point[1] + i_1 < origin_world.shape[0]:
                                    if self.world[dense_point[1] + i_1][dense_point[0] + i_0] == self.map_color["obstacle"]:
                                        source_curve.append([dense_point[0] + i_0, dense_point[1] + i_1])
            # print("curve_dilatation: ",len(curve_dilatation), " --->   source_curve: ", len(source_curve))
            # err_y = util.gauss_noise(0,2)
            # err_x = util.gauss_noise(0,2)
            # err_theta = self.slamErr_angular * np.random.random()
            # print("-> source_curve2: ", len(source_curve))
            err_y =10 * np.random.random()
            err_x =10 * np.random.random()
            err_theta = util.deg2rad(30) * (np.random.random() - 0.5)
            new_source_curve = []
            slam_occup_num = 0
            trans_ppp_index = np.random.choice([0,len(source_curve)-1])
            if source_curve == []:
                stop_times = stop_times + 1 
                continue
            trans_ppp = source_curve[trans_ppp_index]
            real_flag = np.random.choice([0,1],p=[0.3,0.7])
            if real_flag == 0:
                for ddd in range(len(source_curve)):
                    to_trans_point = source_curve[ddd]
                    # print("to_trans_point: ", to_trans_point)
                    rrrrr2 = util.transform_coord(np.array([to_trans_point[1]]),np.array([to_trans_point[0]]), np.array([trans_ppp[1],trans_ppp[0]]),np.array([err_y,err_x,err_theta]))
                    new_point = [int(np.round(rrrrr2[1][0])), int(np.round(rrrrr2[0][0]))]
                    new_point_py = Point(new_point[0], new_point[1])
                    if new_point_py.within(Polygon(_room_boundary_)) == False and new_point_py.within(Polygon(_room_boundary_).boundary) == False: #这个地方可以再考虑下有没有必要把旋转限制在此房间内
                        slam_occup_num = len(source_curve)
                        break
                    if new_point[0] < 0 or new_point[0] >= origin_world.shape[1]:
                        slam_occup_num = len(source_curve)
                        break
                    if new_point[1] < 0 or new_point[1] >= origin_world.shape[0]:
                        slam_occup_num = len(source_curve)
                        break
                    if origin_world[new_point[1]][new_point[0]] == self.map_color["obstacle"]:
                        slam_occup_num = slam_occup_num + 1
                    new_source_curve.append(new_point)
            else:
                # print("miji")
                new_source_curve_original = []
                for ddd in range(len(source_curve)):
                    to_trans_point = source_curve[ddd]
                    # print("to_trans_point: ", to_trans_point)
                    rrrrr2 = util.transform_coord(np.array([to_trans_point[1]]),np.array([to_trans_point[0]]), np.array([trans_ppp[1],trans_ppp[0]]),np.array([err_y,err_x,err_theta]))
                    new_point = [int(np.round(rrrrr2[1][0])), int(np.round(rrrrr2[0][0]))]
                    new_point_py = Point(new_point[0], new_point[1])
                    if new_point_py.within(Polygon(_room_boundary_)) == False and new_point_py.within(Polygon(_room_boundary_).boundary) == False: #这个地方可以再考虑下有没有必要把旋转限制在此房间内
                        slam_occup_num = len(source_curve)
                        break
                    if new_point[0] < 0 or new_point[0] >= origin_world.shape[1]:
                        slam_occup_num = len(source_curve)
                        break
                    if new_point[1] < 0 or new_point[1] >= origin_world.shape[0]:
                        slam_occup_num = len(source_curve)
                        break
                    if origin_world[new_point[1]][new_point[0]] == self.map_color["obstacle"]:
                        slam_occup_num = slam_occup_num + 1
                    new_source_curve_original.append(new_point)
                if len(new_source_curve_original) == 0 or slam_occup_num*1.0/len(new_source_curve_original) > 0.3:
                    pass
                else:
                    for i in range(len(new_source_curve_original)):
                        if i < len(new_source_curve_original) - 1:
                            if [new_source_curve_original[i][0], new_source_curve_original[i][1]] not in new_source_curve:
                                new_source_curve.append([new_source_curve_original[i][0], new_source_curve_original[i][1]])
                            theta_w = math.atan2(new_source_curve_original[i+1][1] - new_source_curve_original[i][1], new_source_curve_original[i+1][0] - new_source_curve_original[i][0])
                            curve_len = math.sqrt((new_source_curve_original[i][1] - new_source_curve_original[i+1][1])*(new_source_curve_original[i][1] - new_source_curve_original[i+1][1]) + (new_source_curve_original[i][0] - new_source_curve_original[i+1][0])*(new_source_curve_original[i][0] - new_source_curve_original[i+1][0]))
                            gap = 0.1
                            grp_num=[]
                            end_gap = 0.1
                            while end_gap < curve_len:
                                grp_num.append(end_gap)
                                end_gap = end_gap + gap
                            for _i_ in grp_num:
                                y_a = int(round(new_source_curve_original[i][1] + _i_ * math.sin(theta_w)))
                                x_a = int(round(new_source_curve_original[i][0] + _i_ * math.cos(theta_w)))
                                if [x_a, y_a] not in new_source_curve:
                                    if x_a < self.world.shape[1] and y_a < self.world.shape[0]:
                                        if x_a >= 0 and y_a >= 0:
                                            new_point_py = Point(x_a, y_a)
                                            if new_point_py.within(Polygon(_room_boundary_)) == True or new_point_py.within(Polygon(_room_boundary_).boundary) == True: #这个地方可以再考虑下有没有必要把旋转限制在此房间内
                                                if origin_world[y_a][x_a] == self.map_color["obstacle"]:
                                                    slam_occup_num = slam_occup_num + 1
                                                new_source_curve.append([x_a, y_a])
                        else:
                            if [new_source_curve_original[i][0], new_source_curve_original[i][1]] not in new_source_curve:
                                new_source_curve.append([new_source_curve_original[i][0], new_source_curve_original[i][1]])
                # print("new_source_curve_original ",len(new_source_curve_original), "->" ,"new_source_curve ", len(new_source_curve))
            
            if len(new_source_curve) == 0 or slam_occup_num*1.0/len(new_source_curve) > 0.3:
                pass
            else:
                # print("-> new_source_curve: ", len(new_source_curve), "  slam_info: ", slam_info)
                for draw_slam_err in new_source_curve:
                    origin_world[draw_slam_err[1]][draw_slam_err[0]] = self.map_color["obstacle"]
                slam_info = slam_info + 1
                # print("boundary_type: ", boundary_type, " slam_error_dilatation: ", slam_error_dilatation)
                # print("slam_noise#", slam_info,"/",slam_noise_num,"stop_times",stop_times)
            stop_times = stop_times + 1 
            # print("1-22")


        """random noise"""
        random_laser_noise_turn_on = np.random.choice([0,1],p=[0.15,0.85])

        random_laser_noise_turn_on = 0


        random_num = np.random.choice([1,2,3,4])
        if random_laser_noise_turn_on == 0:
            random_num = 0
        random_info = 0
        point_num = len(_room_boundary_)
        len_list = [len(_room_boundary_)]

        for ppp_ in B_type_boundary:
            point_num = len(ppp_) + point_num
            len_list.append(len(ppp_))
        random_range_min = min(5,point_num/10.0)
        random_range_max = point_num/2.0
        if point_num < 5:
            random_num = 0
        stop_times = 0
        while random_info < random_num and stop_times < 1000:
            points_source = []
            start_random_p = int((point_num -1) * np.random.random())
            if random_range_max < random_range_min:
                width_random = random_range_max
            else:
                width_random = random_range_min + (random_range_max - random_range_min) * np.random.random()
            width_random = int(round(width_random))
            for target_p in range(start_random_p, start_random_p + width_random):
                end_index_flag = False
                while end_index_flag == False:
                    for len_len in range(len(len_list)):
                        if target_p >= len_list[len_len]:
                            target_p = target_p - len_list[len_len]
                        else:
                            if len_len == 0:
                                points_source.append(_room_boundary_[target_p])
                            else:
                                points_source.append(B_type_boundary[len_len - 1][target_p])
                            end_index_flag = True
                            break
            static_test =0
            for tt_pp in points_source:
                if origin_world[tt_pp[1]][tt_pp[0]] == self.map_color["obstacle"]:
                    if np.random.choice([0,1],p=[0.6,0.4]) == 1:
                        data_index = [[0,1],[1,0],[0,-1],[-1,0],[1,-1],[-1,1],[-1,-1],[1,1]]
                        np.random.shuffle(np.array(data_index))
                        p_num = np.random.choice([1,2,3,4],p=[0.7,0.2,0.05,0.05])
                        _p_num_ = 0
                        for data_p in data_index:
                            if (tt_pp[0]+data_p[0] >= 0 and tt_pp[1]+data_p[1] >=0) and (tt_pp[0]+data_p[0]<origin_world.shape[1] and tt_pp[1]+data_p[1] < origin_world.shape[0]):
                                data_p_polygon = Point(tt_pp[0]+data_p[0],tt_pp[1]+data_p[1])
                                if data_p_polygon.within(Polygon(_room_boundary_)) == True or data_p_polygon.within(Polygon(_room_boundary_).boundary) == True:
                                    if origin_world[tt_pp[1]+data_p[1]][tt_pp[0]+data_p[0]] == self.map_color["free"]:
                                        origin_world[tt_pp[1]+data_p[1]][tt_pp[0]+data_p[0]] = self.map_color["obstacle"]
                                        _p_num_ = _p_num_ + 1
                                        static_test = static_test + 1
                                        if _p_num_ == p_num:
                                            break
            print("Random Noise--> width_random: {}, points_source: {}, static_test: {}, random_range_max: {}, random_range_min: {}".format(width_random, len(points_source),static_test, random_range_max, random_range_min))
            random_info = random_info + 1
            stop_times = stop_times + 1

        """laser noies 1-2"""
        laser_noise_num = np.random.choice([0,1,2,3])
        if laser_noise_turn_on == 0:
            laser_noise_num = 0
        for ___ in range(laser_noise_num):
            obs_y = int(np.random.randint(min_y_l,max_y_l))
            obs_x = int(np.random.randint(min_x_l,max_x_l))
            laser_center = Point(obs_x, obs_y)
            # print("len of _room_boundary_: ", len(_room_boundary_))
            area_room = Polygon(_room_boundary_)
            while origin_world[obs_y][obs_x] == self.map_color["obstacle"] or laser_center.within(area_room)==False:
                obs_y = int(np.random.randint(min_y_l,max_y_l))
                obs_x = int(np.random.randint(min_x_l,max_x_l))
                laser_center = Point(obs_x, obs_y)
            obs_theta = np.random.random() * math.pi * 2
            lasers_info = 0
            stop_times = 0
            lasers = np.random.choice([1,2,3,4,5,6])
            while lasers_info < lasers and stop_times < 1000:
                if stop_times == 0:
                    ttheta = obs_theta
                    # lasers_info.append(obs_theta)
                else:
                    ttheta = obs_theta + ((np.random.random()-0.5) * 2 * math.pi /4)
                    # lasers_info.append(obs_theta + ((np.random.random()-0.5) * 2 * math.pi /4))
                length_laser = int(np.random.random()*self.laser_range/2.0 + self.laser_range/2.0)
                len_seg = [i for i in range(length_laser+1)]
                num_cross = 0
                num_black = 0
                _laser_temp = []
                for point in len_seg:
                    y_ppp = int(round(obs_y - point * math.sin(ttheta)))
                    x_ppp = int(round(obs_x + point * math.cos(ttheta)))
                    if (y_ppp >= 0 and y_ppp < origin_world.shape[0] ) and (x_ppp>=0 and x_ppp < origin_world.shape[1]):
                        if [x_ppp, y_ppp] not in _laser_temp:
                            if [x_ppp, y_ppp] in all_point_lasers:
                                num_cross = num_cross + 1
                            if origin_world[y_ppp][x_ppp] == self.map_color["obstacle"]:
                                num_black = num_black + 1
                            _laser_temp.append([x_ppp, y_ppp])
                if num_cross < 3 and num_black > 0:
                    all_point_lasers = all_point_lasers + _laser_temp
                    for change_point in _laser_temp:
                        origin_world[change_point[1]][change_point[0]] = self.map_color["free"]
                    # print("laser_noise 1 #", lasers_info,"/",___,"stop_times",stop_times)
                    lasers_info = lasers_info + 1
                stop_times = stop_times + 1
        # print("1-20")
        # self.show_contour_pic(origin_world, [], 'noise origin room-0',stop=False)



        return 0, 0, slam_info

    def laser_contour_noise(self, origin_world,min_y_l, max_y_l, min_x_l, max_x_l, _room_boundary_, laser_his, flag=True):
        laser_noise_num = np.random.choice([0,1,2,3])
        for ___ in range(laser_noise_num):
            obs_y = int(np.random.randint(min_y_l,max_y_l))
            obs_x = int(np.random.randint(min_x_l,max_x_l))
            laser_center = Point(obs_x, obs_y)
            # print("len of _room_boundary_: ", len(_room_boundary_))
            area_room = Polygon(_room_boundary_)
            while origin_world[obs_y][obs_x] == self.map_color["obstacle"] or laser_center.within(area_room)==False:
                obs_y = int(np.random.randint(min_y_l,max_y_l))
                obs_x = int(np.random.randint(min_x_l,max_x_l))
                laser_center = Point(obs_x, obs_y)
            obs_theta = np.random.random() * math.pi * 2
            lasers_info = 0
            stop_times = 0
            lasers = np.random.choice([1,2,3,4,5])
            while lasers_info < lasers and stop_times < 1000:
                if stop_times == 0:
                    ttheta = obs_theta
                    # lasers_info.append(obs_theta)
                else:
                    ttheta = obs_theta + ((np.random.random()-0.5) * 2 * math.pi /4)
                    # lasers_info.append(obs_theta + ((np.random.random()-0.5) * 2 * math.pi /4))
                length_laser = int(np.random.random()*self.laser_range/2.0 + self.laser_range/2.0)
                len_seg = [i for i in range(length_laser+1)]
                num_cross = 0
                num_black = 0
                _laser_temp = []
                for point in len_seg:
                    y_ppp = int(round(obs_y - point * math.sin(ttheta)))
                    x_ppp = int(round(obs_x + point * math.cos(ttheta)))
                    if (y_ppp >= 0 and y_ppp < origin_world.shape[0] ) and (x_ppp>=0 and x_ppp < origin_world.shape[1]):
                        if [x_ppp, y_ppp] not in _laser_temp:
                            if [x_ppp, y_ppp] in laser_his:
                                num_cross = num_cross + 1
                            if origin_world[y_ppp][x_ppp] == self.map_color["obstacle"]:
                                num_black = num_black + 1
                            _laser_temp.append([x_ppp, y_ppp])
                if num_cross < 3 and num_black > 0:
                    laser_his = laser_his + _laser_temp
                    for change_point in _laser_temp:
                        origin_world[change_point[1]][change_point[0]] = self.map_color["free"]
                    # print("laser_noise 1 #", lasers_info,"/",___,"stop_times",stop_times)
                    lasers_info = lasers_info + 1
                stop_times = stop_times + 1
        


    def _slam_error_value(self, y_coord, x_coord, y_coord2, x_coord2):
        #_slam_error与_laser_noise过程不同，结果差不多。
        # _laser_noise是依照扫描射线设置误差，使得射线上的点扫偏了，扫到了不正确的位置。
        #而_slam_error则是对整个扫描区域y_noise_rangeCoordMat, x_noise_rangeCoordMat进行相同的旋转与偏移。
        # 旋转偏移的具体过程是按照机器人当前的朝向（即依据局部坐标系）对环境结构中的点（也就是扫描点）进行绕机器人旋转与平移。
        #_laser_noise是根据激光扫描射线制造误差，_slam_error是整体扫描区域制造误差，结果都是将环境结构中的点（self.world）映射到了slam地图中的错误位置

        err_y = util.gauss_noise() * self.slamErr_linear
        err_x = util.gauss_noise() * self.slamErr_linear
        err_theta = util.gauss_noise() * self.slamErr_angular

        """ rotate y_coord & x_coord by err_theta """
        (y_coord_err, x_coord_err)= util.transform_coord(y_coord,x_coord, self.robotPose, np.array([err_y,err_x,err_theta]))
        #shape of np.array([err_y,err_x,err_theta]):(3,)是个三个数的行向量，也就是对所有的y_coord,x_coord点施加相同的噪声
        """ check for index where the coord are within bound """
        inBound_ind= util.within_bound(np.array([y_coord_err,x_coord_err]),self.world.shape)
        x_err_coord= x_coord_err[inBound_ind]
        y_err_coord= y_coord_err[inBound_ind]



        (y_coord_err2, x_coord_err2)= util.transform_coord(y_coord2,x_coord2, self.robotPose, np.array([err_y,err_x,err_theta]))
        #shape of np.array([err_y,err_x,err_theta]):(3,)是个三个数的行向量，也就是对所有的y_coord,x_coord点施加相同的噪声
        """ check for index where the coord are within bound """
        inBound_ind2= util.within_bound(np.array([y_coord_err2,x_coord_err2]),self.world.shape)
        x_err_coord2= x_coord_err2[inBound_ind2]
        y_err_coord2= y_coord_err2[inBound_ind2]
        # x_coord = x_coord.reshape(x_coord.shape[0], 1)[inBound_ind]
        # y_coord = y_coord.reshape(y_coord.shape[0], 1)[inBound_ind]

        return y_err_coord, x_err_coord, y_err_coord2, x_err_coord2




    def _laser_slam_error(self, y_rangeCoordMat, x_rangeCoordMat, y_coord, x_coord, b):
        [y_noise_rangeCoordMat, x_noise_rangeCoordMat, y_coord, x_coord] = self._laser_noise(y_rangeCoordMat, x_rangeCoordMat, y_coord, x_coord, b)
        [y_err_coord, x_err_coord, y_noise_ind, x_noise_ind] = self._slam_error(y_noise_rangeCoordMat, x_noise_rangeCoordMat)

        inBound_ind= util.within_bound(np.array([y_err_coord,x_err_coord]),self.world.shape)
        y_coord = y_coord[inBound_ind]
        x_coord = x_coord[inBound_ind]
        return y_err_coord, x_err_coord, y_coord, x_coord



    def _laser_slam_error_segmentation(self, y_rangeCoordMat, x_rangeCoordMat,obstacle_ind):
       
        y_noise_free, x_noise_free, y_noise_obs, x_noise_obs =self._laser_noise_for_value(y_rangeCoordMat, x_rangeCoordMat,obstacle_ind)

        
        self.slamMap[y_noise_free, x_noise_free]= self.map_color['free']
        self.slamMap[y_noise_obs, x_noise_obs]=self.map_color['obstacle']


    def _build_map_with_rangeCoordMat(self, y_rangeCoordMat, x_rangeCoordMat):
        # Round y and x coord into int
        y_rangeCoordMat = (np.round(y_rangeCoordMat)).astype(np.int)
        x_rangeCoordMat = (np.round(x_rangeCoordMat)).astype(np.int) #（180，41）
        # print("shape of y_rangeCoordMat and x_rangeCoordMat -2: ", y_rangeCoordMat.shape, x_rangeCoordMat.shape,len(y_rangeCoordMat))

        """ Check for index of y_mat and x_mat that are within the world """
        inBound_ind= util.within_bound(np.array([y_rangeCoordMat, x_rangeCoordMat]), self.world.shape)
        # print("shape of inBound_ind: ", inBound_ind.shape) #（180，41）

        """ delete coordinate that are not within the bound """
        outside_ind = np.argmax(~inBound_ind, axis=1)#按照每个扫描射线找到观测区域的地图边界点(利用多个相同最大值取第一个索引)，代表边界外的第一个点的索引，是一个一维矩阵对应每个方向上边界外第一个点的索引
        #outside_ind == 0的含义：该射线上全是边界内的点，或者全是边界外的点或者第一个点是边界外的点。但是由于第一个点代表机器人，所以不可能是边界外点，所以不可能刚出现后两种情况
        # print("shape of outside_ind: ", outside_ind.shape) #（180，）
        ok_ind = np.where(outside_ind == 0)[0] #=0说明在该扫描射线方向上没有出现碰到地图边界的情况，ok_ind代表没有超出边界的射线的索引，应该是一个一维向量
        # print("shape of ok_ind: ", ok_ind.shape) #（180，）
        need_amend_ind = np.where(outside_ind != 0)[0]#说明在该扫描射线方向上出现碰到地图边界的情况，该方向需要进一步被处理，代表存在超出边界的射线的索引，应该是一个一维向量是ok_ind的补集
        outside_ind = np.delete(outside_ind, ok_ind) #只保留需要删除的扫描射线方向上的边界索引，是一个二维矩阵，第一个维是射线序列，第二维只有一个数是边界外第一个点的索引，但是第一维中仅包含需要删除的扫描射线

        inside_ind = np.copy(outside_ind)
        inside_ind[inside_ind != 0] -= 1 #需要删除的扫描射线方向上的边界索引减一，剥离边界，但是按理说inside_ind这里就不会等于0了，因为已经删除了等于0的射线
        bound_ele_x = x_rangeCoordMat[need_amend_ind, inside_ind] #每个射线上最外一个地图内有效点的x值
        bound_ele_y = y_rangeCoordMat[need_amend_ind, inside_ind]

        count = 0
        for i in need_amend_ind:
            x_rangeCoordMat[i, ~inBound_ind[i,:]] = bound_ele_x[count] #把每个射线上地图外的所有点的x换成，该射线上最外一个地图内有效点的x值。没有把地图外的点直接删掉而是换掉应该是处于统一计算的考虑
            y_rangeCoordMat[i, ~inBound_ind[i,:]] = bound_ele_y[count]
            count += 1

        """ find obstacle along the laser range """
        obstacle_ind = np.argmax(self.world[y_rangeCoordMat, x_rangeCoordMat] == self.map_color['obstacle'], axis=1)#找到每个射线上第一个障碍物(利用多个相同最大值取第一个索引),是一个一维矩阵对应每个方向上第一个障碍物的索引
        #obstacle_ind == 0的含义：该射线上全是非障碍物的点，或者全是障碍物或者第一个点是障碍物。但是由于第一个点代表机器人，不可能是障碍物，所以不可能刚出现后两种情况
        obstacle_ind[obstacle_ind == 0] = x_rangeCoordMat.shape[1] #没有障碍物的扫描射线上，索引置为该射线的长度（比实际有效索引大一），弥补仅利用argmax求边界的不足
        #至此，obstacle_ind，代表每个射线上障碍物的索引像素位置，应该是行向量

        # print("shape of obstacle_ind: ",obstacle_ind.shape)
        # print(obstacle_ind)
        # print("self.laser_noiseSigma: ", self.laser_noiseSigma)
        # print("shape of obstacle_ind 1: ", obstacle_ind.shape)


        """ generate a matrix of [[1,2,3,...],[1,2,3...],[1,2,3,...],...] for comparing with the obstacle coord """
        bx = np.arange(x_rangeCoordMat.shape[1]).reshape(1, x_rangeCoordMat.shape[1]) #和self.radius_vect一样是行向量
        by = np.ones((x_rangeCoordMat.shape[0], 1))
        b = np.matmul(by, bx) #b:列数：扫描射线，行：1.,2,3,4...

        #这里有个潜在的问题，如果某条射线上进行了超出边界以外的部分被同化为边界内最后一个坐标，
        # 而恰好整个射线上点没有障碍物满足obstacle_ind == 0，这时候b的有效值就会被重置为x_rangeCoordMat.shape[1]，导致超出边界,虽然b此时超出了边界，但超出边界的点的坐标已经被替换

        """ get the coord that the robot can percieve (ignore pixel beyond obstacle) """
        b = b <= obstacle_ind.reshape(obstacle_ind.shape[0], 1) #把obstacle_ind行向量变成列向量，然后根据obstacle_ind提供的障碍物索引，把那些每个射线上障碍物之前的空闲部分筛选出来，形成bool矩阵（包括第一个障碍物像素，即obstacle_ind所提供的坐标）
        # b就是最终形成的扫描到的实际空闲区域和障碍物边界的掩码
        y_coord = y_rangeCoordMat[b]
        x_coord = x_rangeCoordMat[b]
        #y_rangeCoordMat可能会有重复点，这个是允许的
        #y_coord和x_coord是激光观测扇形范围中扫描到的空闲区域以及扫描到的最近障碍物的所有点的坐标值
        #如果不加噪声，y_coord和x_coord所锁定的点就是需要更新到slam地图上的内容


        
        # print("shape of y_coord: ",y_coord.shape)
        # print("shape of b: ",b.shape, len(b[0]),len(b[3]),len(b[5]),len(b[12]))
        # # 搜索房间区域顶点
        # y_last_ = y_coord[:,-1]
        # x_last_ = x_coord[:,-1]
        # if self.world[y_last_,x_last_] == self.map_color['obstacle']:
        #     pass
        # else:
        #     pass
        


        # """ no slam error """
        # self.slamMap[y_coord, x_coord] = self.world[y_coord, x_coord]



        """segmentation noise"""
        if self.scaning_noise_type == 1:
            self._laser_slam_error_segmentation(y_rangeCoordMat,x_rangeCoordMat,obstacle_ind)
        else:
            self.slamMap[y_coord, x_coord] = self.world[y_coord, x_coord]

        # """ laser noise """
        # # [y_noise_ind,x_noise_ind, y_coord,x_coord]= self._laser_noise(y_rangeCoordMat,x_rangeCoordMat,y_coord,x_coord,b)
        # # self.slamMap[y_noise_ind,x_noise_ind]= self.world[y_coord,x_coord]
        # #

        # """ slam matching error """
        # # [y_err_ind,x_err_ind, y_coord,x_coord] = self._slam_error(y_coord,x_coord)
        # # self.slamMap[y_err_ind,x_err_ind]= self.world[y_coord,x_coord]

        # """ laser noise + slam matching error """
        # # [y_all_noise,x_all_noise, y_coord,x_coord]= self._laser_slam_error(y_rangeCoordMat,x_rangeCoordMat,y_coord,x_coord,b)
        # # self.slamMap[y_all_noise,x_all_noise]= self.world[y_coord,x_coord]
        # #把y_coord,x_coord提供的正确坐标的点对应的值赋到self.slamMap中的错误位置





        """ dilate/close to fill the holes """
        # self.dslamMap= cv2.morphologyEx(self.slamMap,cv2.MORPH_CLOSE,np.ones((3,3)))
        self.dslamMap= cv2.dilate(self.slamMap, np.ones((3,3)), iterations=1)
        return self.slamMap

    def build_map(self, pose):
        """ build perceived map based on robot position and its simulated laser info
        pose: [y;x;theta] in pixel in img coord | robotPose= pose"""
        """ input pose can be in decimal place, it will be rounded off in _build_map_with_rangeCoordMat """

        self.robotPose= pose
        """ find the coord matrix that the laser cover """
        angles= pose[2] + self.angles_vect
        y_rangeCoordMat= pose[0] - np.matmul(np.sin(angles), self.radius_vect)
        x_rangeCoordMat= pose[1] + np.matmul(np.cos(angles), self.radius_vect)

        #此处y的计算是减号是因为self.robotPose中机器人朝向坐标系定义。
        #本程序认为机器人朝上时是90度（即，y轴正方向），而机器人的像素坐标是朝下是90度（即，y轴正方向）。而朝向坐标系和像素坐标系的x方向是一样的

        # print("shape of pose", len(pose),pose.shape)
        # print("shape of self.angles_vect", self.angles_vect.shape, len(self.angles_vect))
        # print("shape of self.radius_vect", self.radius_vect.shape, len(self.radius_vect))
        # print("param: ",self.config["laser"]["range"], self.m2p)
        # print("shape of y_rangeCoordMat and x_rangeCoordMat: ", y_rangeCoordMat.shape, x_rangeCoordMat.shape,len(y_rangeCoordMat))


        # the shape of angles: array([[1],[2],[3],[4]])
        # the shape of self.radius_vect: array([[1, 2, 3, 4]])
        # the shape of y_rangeCoordMat,x_rangeCoordMat: (180,41)
        # y_rangeCoordMat,x_rangeCoordMat对激光扫描射线按照像素进行了逐点采样
        self._build_map_with_rangeCoordMat(y_rangeCoordMat,x_rangeCoordMat)
        # the shape of y_rangeCoordMat and x_rangeCoordMat: array([[1],[2],[3],[4]])
        return self.slamMap



    def moveRobot(self, moveAction):
        """ move robot with moveAction with forward | left | right """
        motion= self.motionChoice[moveAction]
        dv= motion[0]*self.stepLength_linear # forward motion
        dtheta= motion[1]*self.stepLength_angular # angular motion

        """ oversample the motion in each step """
        # sampleNo= 2
        samplePixel= 7
        sampleRad= np.pi/180*10
        sampleNo= np.max([np.abs(dv)*1.0/samplePixel, np.abs(dtheta)*1.0/sampleRad])

        moveLength_step= dv*1./sampleNo
        moveLength_total= 0
        i=0
        while(i<sampleNo):
            # print(self.robotPose)
            theta= self.robotPose[2] + dtheta*1.0/sampleNo
            theta= np.arctan2(np.sin(theta),np.cos(theta))

            # check if remaining step < moveLength_step, if yes, just move the remaining length instead of the whole moveLength_step
            remain_length= self.stepLength_linear-moveLength_total
            if 0< remain_length and remain_length <moveLength_step:
                moveLength_step= self.stepLength_linear-moveLength_total

            y= self.robotPose[0] - np.sin(theta)*moveLength_step
            x= self.robotPose[1] + np.cos(theta)*moveLength_step
            targetPose= np.array([y,x,theta])

            # check if robot will crash on obstacle or go out of bound
            if self.robotCrashed(targetPose):
                self.robotCrashed_flag= True
                # print("Robot crash")
                return False

            if moveAction == "forward":
                self.traj.append([int(x), int(y)])  # only save distince pts

            # build map on the targetPose
            self.build_map( targetPose )
            i=i+1
            moveLength_total+= samplePixel

            #test 
            # print("current pose of robot: ", y,x,theta*180/math.pi)

        return True


    def aground_test(self):
        for an_action in ['forward', 'left', 'right']:
            """ move robot with moveAction with forward | left | right """
            motion= self.motionChoice[an_action]
            dv= motion[0]*self.stepLength_linear # forward motion
            dtheta= motion[1]*self.stepLength_angular # angular motion

            """ oversample the motion in each step """
            # sampleNo= 2
            samplePixel= 7
            sampleRad= np.pi/180*10
            sampleNo= np.max([np.abs(dv)*1.0/samplePixel, np.abs(dtheta)*1.0/sampleRad])

            moveLength_step= dv*1./sampleNo
            moveLength_total= 0
            i=0
            crashed_f = False
            while(i<sampleNo):
                # print(self.robotPose)
                theta= self.robotPose[2] + dtheta*1.0/sampleNo
                theta= np.arctan2(np.sin(theta),np.cos(theta))

                # check if remaining step < moveLength_step, if yes, just move the remaining length instead of the whole moveLength_step
                remain_length= self.stepLength_linear-moveLength_total
                if 0< remain_length and remain_length <moveLength_step:
                    moveLength_step= self.stepLength_linear-moveLength_total

                y= self.robotPose[0] - np.sin(theta)*moveLength_step
                x= self.robotPose[1] + np.cos(theta)*moveLength_step
                targetPose= np.array([y,x,theta])

                # check if robot will crash on obstacle or go out of bound
                if self.robotCrashed(targetPose):
                    crashed_f = True
                    break
                i=i+1
                moveLength_total+= samplePixel
            if crashed_f == False:
                return True
        return False


    def world2state(self):
        # state= cv2.resize(self.slamMap, self.state_size, interpolation=cv2.INTER_LINEAR)
        state= self.slamMap.copy()
        # draw robot position on state
        cv2.circle(state, (int(self.robotPose[1]), int(self.robotPose[0])), self.robotRadius, 50, thickness=-1)

        # draw robot orientation heading on state
        headRadius = np.ceil(self.robotRadius/3.).astype(np.int)
        headLen = self.robotRadius + headRadius
        # orientPt = util.transform_coord(self.robotPose[0], self.robotPose[1], self.robotPose, np.array([0, headLen, 0]))
        # cv2.circle(state, (orientPt[1],orientPt[2]), headRadius, 50, thickness=-1)
        head_y = self.robotPose[0] - np.sin(self.robotPose[2]) * headLen
        head_x = self.robotPose[1] + np.cos(self.robotPose[2]) * headLen
        cv2.circle(state, (int(head_x), int(head_y)), headRadius, 50, thickness=-1)

        if not self.is_exploration:
            """Change color for known environment navigation"""
            state[state == self.map_color['free']] = 255
            state[state == self.map_color['obstacle']] = 0
        return state



    def world2state_large_scale(self):
        # state= cv2.resize(self.slamMap, self.state_size, interpolation=cv2.INTER_LINEAR)
        state= self.slamMap.copy()
        # draw robot position on state
        cv2.circle(state, (int(self.robotPose[1]), int(self.robotPose[0])), self.robotRadius, 50, thickness=-1)

        # draw robot orientation heading on state
        headRadius = np.ceil(self.robotRadius/3.).astype(np.int)
        headLen = self.robotRadius + headRadius
        # orientPt = util.transform_coord(self.robotPose[0], self.robotPose[1], self.robotPose, np.array([0, headLen, 0]))
        # cv2.circle(state, (orientPt[1],orientPt[2]), headRadius, 50, thickness=-1)
        head_y = self.robotPose[0] - np.sin(self.robotPose[2]) * headLen
        head_x = self.robotPose[1] + np.cos(self.robotPose[2]) * headLen
        cv2.circle(state, (int(head_x), int(head_y)), headRadius, 50, thickness=-1)


        flag_0, state_0, ____ = self.target_image_generation(state, [], other_grid_color=[50])
        if flag_0 == True:
            state = state_0
        else:
            state = np.ones([self.out_height,self.out_width])*self.map_color["uncertain"]

        if not self.is_exploration:
            """Change color for known environment navigation"""
            state[state == self.map_color['free']] = 255
            state[state == self.map_color['obstacle']] = 0
        return state




    def robotCrashed(self, pose):
        if ~util.within_bound(pose, self.world.shape, self.robotRadius):
            return True

        py= np.round(pose[0]).astype(int)
        px= np.round(pose[1]).astype(int)
        r= self.robotRadius

        if ~util.within_bound(np.array([py,px,r]), self.world.shape, self.robotRadius):
            return True # Prevent out of range after rounding

        # make a circle patch around robot location and check if there is obstacle pixel inside the circle
        robotPatch, _ = util.make_circle(r,1)
        worldPatch= self.world[py-r:py+r+1, px-r:px+r+1]

        # print("shape of worldPatch, robotPatch: ", worldPatch.shape,robotPatch.shape)
        # print("pose robot crashed: ", pose)
        worldPatch= worldPatch*robotPatch

        #ValueError: operands could not be broadcast together with shapes (8,9) (9,9)
        #I think the hidden bug here comes from using decimal coordinates to judge the extent 
        # and integer coordinates to calculate the matrix
        # shape of worldPatch, robotPatch:  (8, 9) (9, 9)
        #pose robot crashed:  [131.52157904  35.55120436  -1.33038286]
        # map_gt.shape:  (136, 235)
        #Robot Radius        |4 pixels 


        return np.sum(worldPatch==self.map_color["obstacle"])!=0


    def get_state(self):
        return self.world2state().copy()
        # return self.slamMap.copy()

    def get_pose(self):
        return self.robotPose.copy()

    def get_crashed(self):
        return self.robotCrashed_flag

    def measure_ratio(self):
        mapped_pixel= np.sum(self.slamMap==self.map_color['free'])
        world_pixel= np.sum(self.world==self.map_color['free'])

        return 1.*mapped_pixel/world_pixel

    def connect_area_slam(self):
        Binary_slam_m = np.zeros_like(self.world,np.uint8)
        Binary_slam_m[self.world == self.map_color['free']] = 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Binary_slam_m, connectivity=8)
        max_area_la = 0
        max_area = 0
        if len(stats) > 1:
            for i in range(1,len(stats)):
                if stats[i][4] > max_area:
                    max_area = stats[i][4]
                    max_area_la = i
        self.connect_area_1 = labels == max_area_la
        # mapped_pixel_1 = np.sum(self.slamMap[labels == max_area_la] != self.map_color['uncertain'])

        show1 = np.ones_like(Binary_slam_m) * self.map_color['obstacle']
        show1[self.connect_area_1] = self.map_color['free']
        # self.show_contour_pic(show1, [], 'connect_area_slam',stop=False)


        dilated_image = np.ones_like(Binary_slam_m,np.uint8) * 255
        dilated_image[self.world == self.map_color['free']] = 0 
        
        dilated_image = cv2.dilate(dilated_image, np.ones((2*self.robotRadius,2*self.robotRadius), np.uint8), iterations=1)

        dilated_image2 = np.zeros_like(dilated_image,np.uint8)
        dilated_image2[dilated_image == 0] = 255
        num_labels, labels2, stats2, centroids = cv2.connectedComponentsWithStats(dilated_image2, connectivity=8)
        max_area_la2 = 0
        max_area2 = 0
        if len(stats2) > 1:
            for i in range(1,len(stats2)):
                if stats2[i][4] > max_area2:
                    max_area2 = stats2[i][4]
                    max_area_la2 = i
        self.connect_area_2 = labels2 == max_area_la2


        show2 = np.ones_like(dilated_image) * self.map_color['obstacle']
        show2[self.connect_area_2] = self.map_color['free']
        # self.show_contour_pic(show2, [], 'connect_area_dil',stop=False)
        return show1, show2

    def search_doorway(self):
        return []

    def training_data_generation(self, data, map_name):
        self.all_dataset_count['images'] = self.all_dataset_count['images'] + 1
        for contours in data:
            polys = Polygon(contours)
            area = int(round(polys.area))
            bounds = polys.bounds 
            x = int(round(bounds[0]))
            y = int(round(bounds[1]))
            w = int(round(bounds[2]-bounds[0]+1))
            h = int(round(bounds[3]-bounds[1]+1))
            self.all_dataset_count['annotations'] = self.all_dataset_count['annotations'] + 1
            contours_new = [[c for b in contours for c in b ]]
            anno = {'segmentation': contours_new, 'area': area,
                    'image_id': self.all_dataset_count['images'], 'bbox': [x, y, w, h],
                    'iscrowd': 0, 'category_id': 1, 'id': self.all_dataset_count['annotations']}
            self.all_dataset_annotions['annotations'].append(anno)
        
        info = {'file_name': map_name,
                'height': 512, 'width': 512, 'id': self.all_dataset_count['images']}
        self.all_dataset_annotions['images'].append(info)



    def save_simple_dataset_json(self, dataset_annotaitons, dataset_images_info, dataset_simple_anno_path):
        categories = [
            {'supercategory': 'none', 'id': 1, 'name': 'room'}
        ]
        palce_instance = {'images': dataset_images_info, 'annotations': dataset_annotaitons, 'categories': categories}

        with open(dataset_simple_anno_path, 'w') as f:
            json.dump(palce_instance, f)




    def simple_image_generation(self, world_i, boundary_points):
        target_map = np.ones(world_i.shape)*self.map_color["uncertain"]
        tp_map = np.zeros_like(world_i, dtype=np.uint8)
        for i in boundary_points:
            # print("error: ", i[1],i[0], tp_map.shape)
            tp_map[i[1],i[0]]=255
        kernel = np.ones((3, 3), dtype=np.uint8)
        tp_map = cv2.dilate(tp_map, kernel, 1)
        target_map[tp_map==255] = self.map_color["obstacle"]
        target_map[world_i==self.map_color["free"]] = self.map_color["free"]
        return target_map

    def Generation_complete_map(self, processed_map):
        Robot_Observation_Poses = []
        point_flag_o = False
        defined_orientations = [0, np.pi, np.pi/2, -np.pi/2]
        for i in range(len(processed_map)):
            for j in range(len(processed_map[i])):
                if processed_map[i][j] == self.map_color["free"]:
                    point_flag_o = False
                    # downsample
                    abandon_p = False
                    for testing_points in self.passable_points:
                        if math.sqrt((testing_points[0]-i)*(testing_points[0]-i)+(testing_points[1]-j)*(testing_points[1]-j)) < 5:
                            abandon_p = True
                            break
                    if abandon_p == True:
                        continue
                    for de_or in defined_orientations:
                        pose_target = np.array([i,j,de_or])
                        if self.robotCrashed(pose_target):
                            continue
                        else:
                            self.build_map(pose_target)
                            point_flag_o = True
                    if point_flag_o == True:
                        self.passable_points.append([i,j])

    def Save_logs_png(self,slamMap, comMap, slamContours, comContours, simMap=[], simContours=[]):
        if self.noise_world_flag == True and self.rim_count == 0:
            self.write_contour_pic(self.origin_world_floor, [], "plane")
            self.write_contour_pic(self.origin_world_floor, self.segmentation_points, "plane_c")
        if self.noise_world_flag == True:
            mod_ = ""
        else:
            mod_ = "p"
        # self.write_contour_pic(self.world, [], mod_+"Env")
        # self.write_contour_pic(self.world, self.segmentation_points_dynamic_shrinking, mod_+"Env_c")
        self.write_contour_pic(slamMap, slamContours, mod_+"slamMap")
        self.write_contour_pic(comMap, comContours, mod_+"comMap")
        # self.write_contour_pic(self.slamMap, self.segmentation_points_slam, mod_+"slamGri")
        # self.write_contour_pic(self.comMap, self.segmentation_points_slam_com, mod_+"comGri")
        if simMap != []:
            self.write_contour_pic(simMap, simContours, "SimMap")
        
    def write_contour_pic(self, pic, points, name='contour', expand=False):
        if expand == True:
            flag_0, pic_ta, points_ta = self.target_image_generation(pic, points)
            if flag_0 == True:
                pic = pic_ta
                points = points_ta
        new_color_pic = np.ones([pic.shape[0], pic.shape[1],3],dtype=np.uint8)*220
        color_all = [[255,0,0],[0,255,0], [0,0,255],[72,61,139],[0,191,255],[70,130,180],[176,196,222],[34,139,34],
                     [107,142,35],[50,205,50],[255,255,0],[255,215,0],[189,183,107],[0, 255, 255],[255, 0, 255],[128, 128, 128]]
        color_nn = 0
        (h0,w0) = pic.shape[:2]
        if len(pic.shape) == 2:
            for i in range(h0):
                for j in range(w0):
                    if pic[i][j] == self.map_color['obstacle']:
                        new_color_pic[i][j] = (0,0,0)
                    if pic[i][j] == self.map_color['free']:
                        new_color_pic[i][j] = (255,255,255)
        else:
            for i in range(h0):
                for j in range(w0):
                    new_color_pic[i][j] = pic[i][j]
        for a_iii in points:
            color_cc = color_all[color_nn]
            for p_iii in range(len(a_iii)):
                if p_iii == (len(a_iii) - 1):
                    cv2.line(new_color_pic, a_iii[p_iii], a_iii[0],color_cc,2)
                else:
                    # print((a_iii[p_iii],a_iii[p_iii + 1]))
                    cv2.line(new_color_pic, a_iii[p_iii], a_iii[p_iii + 1],color_cc,2)
            color_nn = color_nn + 1
            if color_nn == len(color_all):
                color_nn = 0
        cv2.imwrite(self.config['logs_pic_dir'] + "/" + name + "_R" +str(self.rim_count) + "_"+self.map_id + '.png', new_color_pic)