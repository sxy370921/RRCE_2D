#!/usr/bin/python
# -*- coding: utf-8 -*-
from dis import dis
from operator import index
from re import L, S
from sys import api_version
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter.scrolledtext import ScrolledText
from  tkinter import messagebox
import tkinter as tk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint
import os
import ruamel_yaml as yaml
import math
import numpy as np
import cv2
import json
import copy
# from pycocotools.coco import COCO


class skip_attribute(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.title('selection')
        self.result = -1
        self.setupUI()

    def setupUI(self):
        row1 = tk.Frame(self)
        row1.pack(fill="x")
        l1 = tk.Label(row1, text="Skip to: ", font=('Times', 16))
        l1.pack(side='top', anchor='sw') 
        row01 = tk.Frame(row1)
        row01.pack(side=tk.TOP, fill="x")
        row001 = tk.Frame(row01, width=40, height=3)
        row001.pack(side=tk.LEFT)
        tk.Button(row01, text='Next', command=self.on_click_region, width=18, height=3, activebackground = 'yellow',font=("Times", "12")).pack(side=tk.LEFT)
        tk.Button(row01, text='Reload', command=self.on_click_object, width=18, height=3, activebackground = 'yellow',font=('Times','12')).pack(side=tk.LEFT)   
        row002 = tk.Frame(row01, width=40, height=3)
        row002.pack(side=tk.LEFT)
        self.protocol("WM_DELETE_WINDOW", self.closing)
        self.mainloop()
    
    def closing(self):
        self.result = -1
        self.quit()
        self.destroy()

    def on_click_region(self):
        self.result = 0
        self.quit()
        self.destroy()

    def on_click_object(self):
        self.result = 1
        self.quit()
        self.destroy()

class picture_display(tk.Toplevel):
    def __init__(self, pic,data):
        super().__init__()
        self.pic = pic
        self.flag = False
        self.data = data
        self.title('picture_display')
        self.color_all = ['r', 'y', 'b', 'm','g', 'c', 'pink', 'peru', 'orange','lime','gold']
        self.setupUI()



    def setupUI(self):
        row01 = tk.Frame(self)  
        row01.pack(side=tk.TOP, fill="x")
        self.pic = Image.fromarray(self.pic)
        color_id = 0

        self.f=plt.figure(3, dpi=100)
        # self.f=plt.figure(1,figsize=(8,12)) #figsize决定显示图像的大小
        self.fig3=plt.subplot(1,1,1)
        self.fig3.imshow(self.pic)
        self.canvs = FigureCanvasTkAgg(self.f, row01)
        self.canvs.draw()
        self.canvs.get_tk_widget().pack(side=tk.LEFT)
        self.f2=plt.figure(4, dpi=140)
        # self.f=plt.figure(1,figsize=(8,12)) #figsize决定显示图像的大小
        self.fig4=plt.subplot(1,1,1)
        self.fig4.imshow(self.pic)
        self.canvs2 = FigureCanvasTkAgg(self.f2, row01)
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        for semantic_area in self.data:
            x = [semantic_area[i][0] for i in range(len(semantic_area))]
            y = [semantic_area[i][1] for i in range(len(semantic_area))]
            # x.append(x[0])
            # y.append(y[0])
            # self.fig4.plot(x, y, color='b', linewidth=0.5)
            self.fig4.scatter(x, y, color=self.color_all[color_id], s = 2,alpha=0.3)
            color_id = color_id + 1
            if color_id == len(self.color_all):
                color_id = 0
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
            # self.fig4.scatter(x, y, color='b', linewidths = 8.0)     
        self.canvs2.draw()
        self.canvs2.get_tk_widget().pack(side=tk.LEFT)
        
        # img_png = ImageTk.PhotoImage(self.pic)
        # label_img = tk.Label(row01, image = img_png)
        # label_img.pack(side=tk.LEFT)
        # im = np.zeros((500,500))
        # im = Image.fromarray(im)
        # img_png2 = ImageTk.PhotoImage(im)
        # label_img2 = tk.Label(row01, image = img_png2)
        # label_img2.pack(side=tk.LEFT)
        row02 = tk.Frame(self)
        row02.pack(side=tk.TOP, fill="x")
        tk.Button(row02, text='OK', command=self.OK, width=18, height=3, activebackground = 'yellow',font=("Times", "12")).pack(side=tk.TOP)
        tk.Button(row02, text='Error', command=self.Error, width=18, height=3, activebackground = 'yellow',font=("Times", "12")).pack(side=tk.TOP)
        self.protocol("WM_DELETE_WINDOW", self.closing)
        self.mainloop()
    
    def get_flag(self):
        # print("flag = ",self.flag)
        return self.flag

    def closing(self):
        self.flag = False
        # print("inter closing")
        # print("flag = ",self.flag)
        plt.close(3)
        plt.close(4)
        self.quit()
        self.destroy()

    def OK(self):
        # print("inter ok")
        self.flag = True
        # print("flag = ",self.flag)
        # print("flag = ",self.flag)
        plt.close(3)
        plt.close(4)
        self.quit()
        self.destroy()
    
    def Error(self):
        # print("inter error")
        self.flag = False
        # print("flag = ",self.flag)
        plt.close(3)
        plt.close(4)
        self.quit()
        self.destroy()


class semantic_annotation():
    def __init__(self):
        # initialize variable

        self.simple_dataset_generate_flag = True
        self.dataset_display_flag = True

        self.flag_load = False
        self.flag_area = False

        self.rewrite = False #
        self.achievement = False

        self.meter2pixel = 20
        self.border_pad = 4
        self.room_max = 100
        self.room_min = 3
        self.out_height = 512
        self.out_width = 512
        self.out_center_height = int(self.out_height//2)
        self.out_center_width = int(self.out_width//2)

        ###
        map_ids_name = 'annotation_list.txt'
        # data_path = "./"
        image_dir_name = './others'
        annotations_dir_name = './others'

        self.curpath = os.path.dirname(os.path.realpath(__file__))
        self.map_ids_path = os.path.join(self.curpath, 'files', map_ids_name)
        self.dataset_simple_anno_path = os.path.join(annotations_dir_name, 'sxy.json')
        self.dataset_simple_image_path = os.path.join(image_dir_name)
        self.existing_path = self.dataset_simple_anno_path
        self.original_map_path = './json'
        self.segmentation_data_path = './output_segmentation_data'
        self.segmentation_data_list_path = './output_segmentation_list'
        self.record_data_path = './output_record_points'
        # print(self.dataset_simple_anno_path,'\n',self.dataset_simple_image_path)
        # if os.path.exists(self.existing_path):
        #     self.coco = COCO(self.existing_path)
        #     self.anns = self.coco.getImgIds()
        self.total_map_ids = list(np.loadtxt(self.map_ids_path, str))
        self.target_map_ids = []
        self.skip_ids = []
        json_name = os.listdir(self.segmentation_data_path)
        # print(json_name)
        tmp_existing_ids = [map_i.split('.')[0] for map_i in json_name]
        if tmp_existing_ids == []:
            self.target_map_ids = self.total_map_ids
        else:
            for i_name in self.total_map_ids:
                if i_name not in tmp_existing_ids:
                    self.target_map_ids.append(i_name)
        # tmp_existing_ids = self.total_map_ids - self.self.target_map_ids
        print("\n=====START UP=====\n")
        print("\n\033[31m@ The map list file processed by this program:@\033[0m")
        print("\n\033[31m@ {} @\033[0m".format(map_ids_name))
        print("total_ids: ", len(self.total_map_ids))
        print("target_ids: ", len(self.target_map_ids))
        print("existing_ids: ", len(tmp_existing_ids))
        # if len(self.target_map_ids) == 0:
        #     print("There are no unannotated maps!!!")
        #     return

        
        # self.existing_map_ids = []
        # self.target_map_ids = list(set(self.total_map_ids) - set(self.existing_map_ids))
        self.saved_map_ids = []
        # self.processed_map_ids = []
        self.unsaved_map_ids = []
        self.unsaved_map_data = {}
        self.unsaved_dataset_image = {}
        self.map_index = 0
        self.dataset_mages_info = []
        self.dataset_annotations = []
        self.dataset_annotations_count = 0
        self.dataset_image_count = 0

        
        self.this_map_id = ''
        self.this_json_data = {}
        self.this_black_contour = []
        self.this_map = np.zeros((500,500))
        self.this_image = Image.fromarray(self.this_map)
        self.this_box = {}
        self.this_dataset_map = np.full((self.out_height,self.out_width,3),220, dtype=np.uint8)
        self.this_dataset_contours = []
        self.this_dataset_map_name = ''
        self.this_offset = {'offset_height':-1,'offset_width':-1}
        self.this_flag_consistent_0=False
        self.this_flag_consistent_1=False
        
        self.test_list = {}
        self.test_list2 = {}
        self.test_list3 = {}

        self.record_data_flag = True
        self.unsaved_record_data = {}
        self.this_record_data = []
        


        # configure GUI
        self.root = tk.Tk()
        self.root.title("semantic annotation")
        self.f=plt.figure(1)
        # self.f=plt.figure(1,figsize=(8,12)) #figsize决定显示图像的大小
        self.fig1=plt.subplot(1,1,1)
        # im = np.zeros((500,500))
        # im = Image.fromarray(self.this_image)
        # im = Image.open(self.map_path)
        # self.fig1.imshow(im, cmap = plt.get_cmap("gray"))
        self.fig1.imshow(self.this_image)
        self.canvs = FigureCanvasTkAgg(self.f, self.root)
        self.canvs.draw()
        self.canvs.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.text = ScrolledText(self.root, width=50, height=30, font=('Times','16'))
        self.text.pack(side=tk.TOP)
        # self.text = tk.Text(self.root, width=40, height=30).pack(side=tk.TOP)
        row01 = tk.Frame(self.root)  # pack的原则是在剩余空间插入插件。Frame可以用来占位,然后再从Frame中布局。这样可以实现更灵活的布局
        row01.pack(side=tk.TOP, fill="x")
        row001 = tk.Frame(row01, width=80, height=3)
        row001.pack(side=tk.LEFT)
        tk.Button(row01, text='(1) Area', command=self.select_area, width=18, height=3, activebackground = 'yellow',font=("Times", "12")).pack(side=tk.LEFT)
        tk.Button(row01, text='(2) Door ', command=self.select_door , width=18, height=3, activebackground = 'yellow',font=('Times','12')).pack(side=tk.LEFT)
        # tk.Button(row01, text='Skip', command=self.skip_error_data, width=18, height=3, activebackground = 'yellow',font=('Times','12')).pack(side=tk.LEFT)
        row03 = tk.Frame(self.root)
        row03.pack(side=tk.TOP, fill="x")
        row003 = tk.Frame(row03, width=80, height=3)
        row003.pack(side=tk.LEFT)
        tk.Button(row03, text='Map Up', command=self.last_map, width=18, height=3, activebackground = 'yellow',font=("Times", "12")).pack(side=tk.LEFT)
        tk.Button(row03, text='File render', command=self.render, width=18, height=3, activebackground = 'yellow',font=('Times','12')).pack(side=tk.LEFT)
        row02 = tk.Frame(self.root)
        row02.pack(side=tk.TOP, fill="x")
        row002 = tk.Frame(row02, width=80, height=3)
        row002.pack(side=tk.LEFT)
        tk.Button(row02, text='Delete', command=self.delete_area, width=18, height=3, activebackground = 'yellow',font=('Times','12')).pack(side=tk.LEFT)
        tk.Button(row02, text='Confirm', command=self.confirm, width=18, height=3, activebackground = 'yellow',font=('Times','12')).pack(side=tk.LEFT)
        row04 = tk.Frame(self.root)
        row04.pack(side=tk.TOP, fill="x")
        row004 = tk.Frame(row04, width=80, height=3)
        row004.pack(side=tk.LEFT)
        tk.Button(row04, text='Skip', command=self.skip_error_data, width=18, height=3, activebackground = 'yellow',font=('Times','12')).pack(side=tk.LEFT)
        tk.Button(row04, text='Exit', command=self.exit, width=18, height=3, activebackground = 'red',font=('Times','12'),fg='red').pack(side=tk.LEFT)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.text.insert(tk.END,'total_ids: {}\ntarget_ids: {}\nexisting_ids: {}\n\n\n'.format(len(self.total_map_ids), len(self.target_map_ids), len(tmp_existing_ids)))
        self.text.see(tk.END)
        self.load_image()
        self.root.mainloop()

    def poly_not_cross_open(self,points,test_point,close_flag=0):
        # Edges formed by adjacent points do not have additional intersects
        # Coordinate values must be integers
        def is_cross(A_start, A_end, B_start, B_end):
            point_is_exist = False
            # x = y = 0
            x1 = A_start[0]
            y1 = A_start[1]
            x2 = A_end[0]
            y2 = A_end[1]
            x3 = B_start[0]
            y3 = B_start[1]
            x4 = B_end[0]
            y4 = B_end[1]

            if (x2 - x1) == 0:
                k1 = None
                b1 = 0
            else:
                k1 = (y2 - y1) * 1.0 / (x2 - x1)
                b1 = y1 - x1 * k1 

            if (x4 - x3) == 0:  
                k2 = None
                b2 = 0
            else:
                k2 = (y4 - y3) * 1.0 / (x4 - x3)  
                b2 = y3  - x3 * k2

            if k1 is None:
                if not (k2 is None):
                    x = x1
                    y = k2 * x1 + b2
                    if (min(y1,y2)<=y<=max(y1,y2)) and (min(x3,x4)<=x<=max(x3,x4)):
                        point_is_exist = True
                else:
                    if (x1 == x3) and ((min(y3,y4)<=min(y1,y2)<=max(y3,y4))or(min(y1,y2)<=min(y3,y4)<=max(y1,y2))or(min(y3,y4)<=max(y1,y2)<=max(y3,y4))or(min(y1,y2)<=max(y3,y4)<=max(y1,y2))):
                        #  the endpoints don't overlap
                        point_is_exist = True
                        

            elif k2 is None:
                x = x3
                y = k1 * x3 + b1
                if (min(y3,y4)<=y<=max(y3,y4)) and (min(x1,x2)<=x<=max(x1,x2)):
                    point_is_exist = True                
            elif k2 == k1:
                if (b1==b2) and ((min(x3,x4)<=min(x1,x2)<=max(x3,x4))or(min(x1,x2)<=min(x3,x4)<=max(x1,x2))or(min(x3,x4)<=max(x1,x2)<=max(x3,x4))or(min(x1,x2)<=max(x3,x4)<=max(x1,x2))):
                    point_is_exist = True  
            else:
                x = (b2 - b1) * 1.0 / (k1 - k2)
                y = k1 * x * 1.0 + b1 * 1.0
                if (min(x1,x2)<=x<=max(x1,x2)) and (min(x3,x4)<=x<=max(x3,x4)):
                    point_is_exist = True  
            return point_is_exist
        def last_point_test(p1,p2,p_test):
            # print("$$$",p1,p2,p_test)
            x1 = p1[0]
            y1 = p1[1]
            x2 = p2[0]
            y2 = p2[1]
            x3 = p_test[0]
            y3 = p_test[1]
            if (x2 - x1) == 0:
                # print("t1")
                if (x3 - x2) == 0:
                    # print("t2")
                    if (y3-y2)*(y1-y2)>=0:
                        # print("t3")
                        return True
            else:
                k=(y2 - y1) * 1.0 / (x2 - x1)
                if ((x3-x2) != 0) and (k==((y3 - y2) * 1.0 / (x3 - x2))):
                    if (x3-x2)*(x1-x2)>=0:
                        return True
            return False



        abnormal_point = []
        for index, i in enumerate(points):
            if (i[0] == test_point[0]) and (i[1] == test_point[1]):
                abnormal_point.append(index)
        
        if len(abnormal_point)>0:
            if (len(abnormal_point) == 1) and (abnormal_point[0]==0):
                if close_flag == 0:
                    print("There are repeating points in the point set-0")
                    return False
            else:
                print("There are repeating points in the point set-1")
                return False
        else:
            if close_flag != 0:
                print("This is not the last point")
                return False
        # The execution can be continued only in the following two cases:
        # 1) len(abnormal_point)=0,close_flag!=0  2)len(abnormal_point) = 1,abnormal_point[0]=0,close_flag != 0
        if len(points)>2:
            # print("tested segments(index):") # test-info
            if close_flag == 0:
                for num in range(len(points)-2):
                    if is_cross(points[num],points[num+1],points[-1],test_point)==True:
                        # print("({},{}) false".format(num,num+1)) # test-info
                        # print("[cross-detail 1: ]", points[num],points[num+1],points[-1],test_point, num, num+1)
                        return False
                    # else:
                    #     print("({},{}) true".format(num,num+1)) # test-info
                if last_point_test(points[len(points)-2],points[len(points)-1],test_point)==True:
                    # print("[cross-detail 2: ]", points[len(points)-2],points[len(points)-1],test_point)
                    # print("last_point_test false") # test-info
                    return False
            else:
                for num in range(1,len(points)-2):
                    if is_cross(points[num],points[num+1],points[-1],test_point)==True:
                        # print("({},{}) false".format(num,num+1)) # test-info
                        # print("[cross-detail 3: ]", points[num],points[num+1],points[-1],test_point, num, num+1)
                        return False
                    # else:
                    #     print("({},{}) true".format(num,num+1)) # test-info
                if last_point_test(points[len(points)-2],points[len(points)-1],test_point)==True:
                    # print("last_point_test false") # test-info
                    # print("[cross-detail 4: ]", points[len(points)-2],points[len(points)-1],test_point)
                    return False
                if last_point_test(points[len(points)-1],test_point,points[1])==True:
                    # print("[cross-detail 5: ]", points[len(points)-1],test_point,points[1])
                    # print("last_point_test2 false") # test-info
                    return False
            return True
        else:
            if close_flag == 0:
                if len(points) == 2:
                    # print("test_len=2") # test-info
                    if last_point_test(points[0],points[1],test_point)==True:
                        # print("test_len=2 false") # test-info
                        return False
                    else:
                        return True
                else:
                    return True
            else:
                print('Two points can not be closed')
                return False

    def segment_orientation(self, a1,a2,b1,b2):
        dx1 = a2[0]-a1[0]
        dy1 = a2[1]-a1[1]
        dx2 = b2[0]-b1[0]
        dy2 = b2[1]-b1[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(angle1 * 180/math.pi)
        # print(angle1)
        angle2 = math.atan2(dy2, dx2)
        angle2 = int(angle2 * 180/math.pi)
        # print(angle2)
        if angle1*angle2 >= 0:
            included_angle = abs(angle1-angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle


    def simple_image_generation(self):
        color_uncertain = 220
        color_obstacle = 0
        color_free = 254
        target_map = np.full((self.out_height,self.out_width,3),color_uncertain, dtype=np.uint8)
        tp_map = np.zeros_like(self.this_map, dtype=np.uint8)
        for i in self.this_black_contour:
            tp_map[i[1],i[0]]=255
        kernel = np.ones((3, 3), dtype=np.uint8)
        tp_map = cv2.dilate(tp_map, kernel, 1)
        tp_map[tp_map==0] = color_uncertain
        tp_map[tp_map==255] = color_obstacle
        tp_map[self.this_map==255] = color_free

        height, width = tp_map.shape[0], tp_map.shape[1]
        center_height = int(tp_map.shape[0]//2)
        center_width = int(tp_map.shape[1]//2)

        if (height<=self.out_height) and (width <= self.out_width):
            offset_height = self.out_center_height - center_height
            offset_width = self.out_center_width - center_width
            for h in range(height):
                    for w in range(width):
                        if (tp_map[h][w] == color_free) or (tp_map[h][w] == color_obstacle):
                            target_map[h+offset_height][w+offset_width]=tp_map[h][w]
        else:
            offset_height = -1
            offset_width = -1
            tk.messagebox.showinfo(title='warning', message='image shape error!')
        self.this_offset['offset_height'] = offset_height
        self.this_offset['offset_width'] = offset_width
        return target_map

    def simple_anno_generation(self,pixels_i=2):
        #目前比较好的参数是pixels_i=2，d_d<3，此时d_d越大比如10越无法描绘边界
        def GeneralEquation(first_x,first_y,second_x,second_y):
            A = first_y - second_y
            B = first_x - second_x
            if B == 0:
                return int(1),0,first_x
            elif A == 0:
                return int(0),0,first_y
            else:
                K = 1.0*A/B
                B = first_y - K * first_x
            
            return int(0),K, B
        def distance(x1,y1,x2,y2):
            return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
        
        print("[......] rendering sampled points")
        contour = copy.deepcopy(self.this_black_contour)
        point_group = self.this_json_data[self.this_map_id]['areas']
        re_areas = []
        for points in point_group:
            re_segment = []
            # statistic_anno_points_num = 0
            #这里的points是带着首尾重复点的
            for i in range(len(points)-1):
                # test_statistic
                num_0_teat = 0
                num_1_teat = 0
                num_2_teat = 0
                num_3_teat = 0
                num_4_teat = 0
                num_5_teat = 0
                len_seg = distance(points[i][0],points[i][1],points[i+1][0],points[i+1][1])
                # len_seg = int(round(len_seg))
                # interval_pixels_len = 10
                interval_pixels_len = pixels_i
                re_segment.append(points[i]) #import
                if len_seg > (1.5 * interval_pixels_len):
                    # print("len_seg", len_seg)
                    sample_num = int(round(len_seg/interval_pixels_len))
                    # sample_num = int(len_seg/interval_pixels_len)
                    # print("sample_num", sample_num)
                    ty,k,b = GeneralEquation(points[i][0],points[i][1],points[i+1][0],points[i+1][1])
                    interval = (points[i+1][ty]-points[i][ty])/float(sample_num)
                    if (points[i+1][ty]-points[i][ty]) >= 0:
                        direction = 1
                    else:
                        direction = -1

                    if ty == 0:
                        pointer_x = points[i][0]
                    else:
                        pointer_y = points[i][1]

                    for __ in range(sample_num-1):
                        if ty == 0:
                            pointer_x = pointer_x+ interval
                            pointer_y = k*pointer_x + b
                            pointer_ty = pointer_x
                        else:
                            pointer_y = pointer_y+ interval
                            pointer_x = b
                            pointer_ty = pointer_y
                        if (direction*points[i+1][ty])>direction*(pointer_ty+direction/2):
                            re_point = -1
                            re_index = -1
                            haha = -1
                            d_min = float('inf')
                            for index, pt in enumerate(contour):
                                if pt not in re_segment:
                                    d_d = distance(pointer_x,pointer_y,pt[0],pt[1])
                                    if d_d <3 and d_d < d_min:
                                        # num_0_teat = num_0_teat + 1
                                        # print("verts, segment, alternative points", [points[i],points[i+1]],(pointer_x,pointer_y), pt )
                                        if (pt[0]!= points[i][0]) or (pt[1]!= points[i][1]):
                                            if (pt[0]!= points[i+1][0]) or (pt[1]!= points[i+1][1]):
                                                haha = 1
                                                if (0 <= self.segment_orientation(points[i],points[i+1],re_segment[-1],pt) < 90.5) and (0 <= self.segment_orientation(points[i],points[i+1],pt,points[i+1]) < 90.5):
                                                    if self.poly_not_cross_open(re_segment, pt) == True:
                                                        if i == (len(points)-2): #这部分是临时加的，因为如果没有这个判断，会导致添加的下一个顶点无法保证与上采样的点不相交。但这样一加计算量增大挺多
                                                            if self.poly_not_cross_open(re_segment+[pt], points[i+1], close_flag=1) == True:
                                                                d_min = d_d
                                                                re_point = pt
                                                                re_index = index
                                                                # num_5_teat = num_5_teat + 1
                                                            else:
                                                                pass
                                                                # num_4_teat = num_4_teat + 1
                                                                # print("[] The point bring bad vertex 1", re_segment[0], re_segment[-1], pt, points[i+1])
                                                        else:
                                                            if self.poly_not_cross_open(re_segment+[pt], points[i+1]) == True:
                                                                d_min = d_d
                                                                re_point = pt
                                                                re_index = index
                                                                # num_5_teat = num_5_teat + 1
                                                            else:
                                                                pass
                                                                # num_3_teat = num_3_teat + 1
                                                                # print("[] The point bring bad vertex 2", re_segment[0], re_segment[-1], pt, points[i+1])
                                                    else:
                                                        # print("verts, segment, alternative points", [points[i],points[i+1]],(pointer_x,pointer_y), pt )
                                                        # print("poly_not_cross_open")
                                                        # num_2_teat = num_2_teat + 1
                                                        pass
                                                        
                                                else:
                                                    # print("verts, segment, alternative points", [points[i],points[i+1]],(pointer_x,pointer_y), pt )
                                                    # print("segment_orientation")
                                                    # num_1_teat = num_1_teat + 1
                                                    pass
                            # print("finded point: ", re_point)

                            if re_index >= 0:
                                re_segment.append(re_point)
                            else:
                                print("No Points ", "in ", [points[i],points[i+1]], "--> {:.3f},{:.3f}".format(pointer_x, pointer_y))
                                if haha == -1:
                                    print("Inappropriate distance")
                # print("[test]  orientation-{}, cross-{}, end_cross-{}, end_cross_0-{}, all-{}, ok-{}".format(num_1_teat, num_2_teat, num_3_teat, num_4_teat, num_0_teat,num_5_teat))
            re_segment.append(points[i+1])
            re_areas.append(re_segment)
            print("room", point_group.index(points)," points: ", len(points), "-->", len(re_segment) )

        return re_areas


    def simple_anno_generation_v2(self,pixels_i=2):
        #目前比较好的参数是pixels_i=2，d_d<3，此时d_d越大比如10越无法描绘边界
        # def GeneralEquation(first_x,first_y,second_x,second_y):
        #     A = first_y - second_y
        #     B = first_x - second_x
        #     if B == 0:
        #         return int(1),0,first_x
        #     elif A == 0:
        #         return int(0),0,first_y
        #     else:
        #         K = 1.0*A/B
        #         B = first_y - K * first_x
        
        #     return int(0),K, B
        def distance(x1,y1,x2,y2):
            return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
        
        print("[......] rendering sampled points")
        print("[......] simple_anno_generation_v2 [{},{}]".format(pixels_i,5))
        contour = copy.deepcopy(self.this_black_contour)
        point_group = self.this_json_data[self.this_map_id]['areas']
        re_areas = []
        for points in point_group:
            re_segment = []
            # statistic_anno_points_num = 0
            #这里的points是带着首尾重复点的
            for i in range(len(points)-1):
                angle_seg = math.atan2(points[i+1][1] - points[i][1], points[i+1][0] - points[i][0])
                len_seg = distance(points[i][0],points[i][1],points[i+1][0],points[i+1][1])
                # len_seg = int(round(len_seg))
                # interval_pixels_len = 10
                interval_pixels_len = pixels_i
                re_segment.append(points[i]) #import
                if len_seg > (1.5 * interval_pixels_len):
                    line_range = np.arange(interval_pixels_len,len_seg,step=interval_pixels_len)
                    line_range = line_range.tolist()
                    for line_vextor in line_range:
                        pointer_x = points[i][0] + math.cos(angle_seg) * line_vextor
                        pointer_y = points[i][1] + math.sin(angle_seg) * line_vextor
                        # if (direction*points[i+1][ty])>direction*(pointer_ty+direction/2):
                        re_point = -1
                        re_index = -1
                        # haha = -1
                        d_min = float('inf')
                        for index, pt in enumerate(contour):
                            if pt not in re_segment:
                                d_d = distance(pointer_x,pointer_y,pt[0],pt[1])
                                if d_d <5 and d_d < d_min:
                                    # num_0_teat = num_0_teat + 1
                                    # print("verts, segment, alternative points", [points[i],points[i+1]],(pointer_x,pointer_y), pt )
                                    if (pt[0]!= points[i][0]) or (pt[1]!= points[i][1]):
                                        if (pt[0]!= points[i+1][0]) or (pt[1]!= points[i+1][1]):
                                            # haha = 1
                                            if (0 <= self.segment_orientation(points[i],points[i+1],re_segment[-1],pt) < 90.5) and (0 <= self.segment_orientation(points[i],points[i+1],pt,points[i+1]) < 90.5):
                                                polygon_valid_contour = re_segment+[pt]+points[i+1:]
                                                if Polygon(polygon_valid_contour).is_valid == True:
                                                    if self.poly_not_cross_open(re_segment, pt) == True:
                                                        if i == (len(points)-2): #这部分是临时加的，因为如果没有这个判断，会导致添加的下一个顶点无法保证与上采样的点不相交。但这样一加计算量增大挺多
                                                            if self.poly_not_cross_open(re_segment+[pt], points[i+1], close_flag=1) == True:
                                                                d_min = d_d
                                                                re_point = pt
                                                                re_index = index
                                                                # num_5_teat = num_5_teat + 1
                                                            else:
                                                                pass
                                                                # num_4_teat = num_4_teat + 1
                                                                # print("[] The point bring bad vertex 1", re_segment[0], re_segment[-1], pt, points[i+1])
                                                        else:
                                                            if self.poly_not_cross_open(re_segment+[pt], points[i+1]) == True:
                                                                d_min = d_d
                                                                re_point = pt
                                                                re_index = index
                                                                # num_5_teat = num_5_teat + 1
                                                            else:
                                                                pass
                                                                # num_3_teat = num_3_teat + 1
                                                                # print("[] The point bring bad vertex 2", re_segment[0], re_segment[-1], pt, points[i+1])
                                                    else:
                                                        # print("verts, segment, alternative points", [points[i],points[i+1]],(pointer_x,pointer_y), pt )
                                                        # print("d_d, d_min, re_point, re_index: ", d_d, d_min, re_point, re_index)
                                                        # print("Cross: ", pt, re_segment[-1])
                                                        # num_2_teat = num_2_teat + 1
                                                        pass
                                                else:
                                                    pass
                                                    
                                            else:
                                                # print("verts, segment, alternative points", [points[i],points[i+1]],(pointer_x,pointer_y), pt )
                                                # print("segment_orientation")
                                                # num_1_teat = num_1_teat + 1
                                                # print("d_d, d_min, re_point, re_index: ", d_d, d_min, re_point, re_index)
                                                # print("orientat: ",points[i],points[i+1],re_segment[-1],pt)
                                                pass
                        # print("finded point: ", re_point)

                        if re_index >= 0:
                            re_segment.append(re_point)
                        else:
                            pass
                            # print("No P ", "in ", [points[i],points[i+1]], "--> {:.3f},{:.3f}".format(pointer_x, pointer_y))
                            # if haha == -1:
                            #     print("Inappropriate distance")
                # print("[test]  orientation-{}, cross-{}, end_cross-{}, end_cross_0-{}, all-{}, ok-{}".format(num_1_teat, num_2_teat, num_3_teat, num_4_teat, num_0_teat,num_5_teat))
            re_segment.append(points[i+1])
            re_areas.append(re_segment)
            print("room", point_group.index(points)," points: ", len(points), "-->", len(re_segment) )

        return re_areas



    def tune_door(self,point_group):
        def GeneralEquation(first_x,first_y,second_x,second_y):
            A = first_y - second_y
            B = first_x - second_x
            if B == 0:
                return int(1),0,first_x
            elif A == 0:
                return int(0),0,first_y
            else:
                K = 1.0*A/B
                B = first_y - K * first_x
            
            return int(0),K, B
        def distance(x1,y1,x2,y2):
            return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
        
        print("[......] tuning new door")
        contour = copy.deepcopy(self.this_black_contour)
        re_areas = []
        re_segment = []
        # statistic_anno_points_num = 0
        #这里的points是带着首尾重复点的
        i = 0
        len_seg = distance(point_group[i][0],point_group[i][1],point_group[i+1][0],point_group[i+1][1])
        # len_seg = int(round(len_seg))
        # interval_pixels_len = 10
        interval_pixels_len = 1
        re_segment.append(point_group[i]) #import
        if len_seg > (1.5 * interval_pixels_len):
            # print("len_seg", len_seg)
            sample_num = int(round(len_seg/interval_pixels_len))
            # sample_num = int(len_seg/interval_pixels_len)
            # print("sample_num", sample_num)
            ty,k,b = GeneralEquation(point_group[i][0],point_group[i][1],point_group[i+1][0],point_group[i+1][1])
            interval = (point_group[i+1][ty]-point_group[i][ty])/float(sample_num)
            if (point_group[i+1][ty]-point_group[i][ty]) >= 0:
                direction = 1
            else:
                direction = -1

            if ty == 0:
                pointer_x = point_group[i][0]
            else:
                pointer_y = point_group[i][1]

            for __ in range(sample_num-1):
                if ty == 0:
                    pointer_x = pointer_x+ interval
                    pointer_y = k*pointer_x + b
                    pointer_ty = pointer_x
                else:
                    pointer_y = pointer_y+ interval
                    pointer_x = b
                    pointer_ty = pointer_y
                if (direction*point_group[i+1][ty])>(direction*pointer_ty+direction/2):
                    re_point = -1
                    re_index = -1
                    haha = -1
                    d_min = float('inf')
                    for index, pt in enumerate(contour):
                        if pt not in re_segment:
                            d_d = distance(pointer_x,pointer_y,pt[0],pt[1])
                            if d_d <3 and d_d < d_min:
                                # print("verts, segment, alternative points", [points[i],points[i+1]],(pointer_x,pointer_y), pt )
                                if (pt[0]!= point_group[i][0]) or (pt[1]!= point_group[i][1]):
                                    if (pt[0]!= point_group[i+1][0]) or (pt[1]!= point_group[i+1][1]):
                                        haha = 1
                                        if 0 <= self.segment_orientation(point_group[i],point_group[i+1],re_segment[-1],pt) < 90.5:
                                            if self.poly_not_cross_open(re_segment, pt) == True:
                                                if self.poly_not_cross_open(re_segment+[pt], point_group[i+1]) == True:
                                                    d_min = d_d
                                                    re_point = pt
                                                    re_index = index
                                                else:
                                                    print("[] The point bring bad vertex 2", re_segment[0], re_segment[-1], pt, point_group[i+1])
                                            else:
                                                # print("verts, segment, alternative points", [points[i],points[i+1]],(pointer_x,pointer_y), pt )
                                                # print("self.poly_not_cross_open(re_segment, pt) == false")
                                                pass
                                                
                                        else:
                                            # print("verts, segment, alternative points", [points[i],points[i+1]],(pointer_x,pointer_y), pt )
                                            # print("self.segment_orientation", self.segment_orientation(points[i],points[i+1],re_segment[-1],pt))
                                            pass
                    # print("finded point: ", re_point)

                    if re_index >= 0:
                        re_segment.append(re_point)
                    # else:
                    #     print("There is no proper point ", "in ", [points[i],points[i+1]])
                    #     if haha == -1:
                    #         print("Inappropriate distance")

        re_segment.append(point_group[i+1])
        print("length of re_segment -- ", len(re_segment))
        dis_tra = 0
        for pn in range(len(re_segment)-1):
            d0_d0 = distance(re_segment[pn][0],re_segment[pn][1],re_segment[pn+1][0],re_segment[pn+1][1])
            if d0_d0 > dis_tra:
                dis_tra = d0_d0
                re_areas = [re_segment[pn], re_segment[pn+1]]

        print("r1: ", re_areas)
        
        for yyy in range(2):
            re_x3i = []
            dis_tra_222 = float('inf')
            tmp_re = re_areas[yyy]
            for x3i in self.this_black_verts:
                if x3i not in re_areas:
                    ddd_ddd9 = distance(tmp_re[0],tmp_re[1],x3i[0], x3i[1])
                    if ddd_ddd9 < 2 and ddd_ddd9 < dis_tra_222:
                        dis_tra_222 = ddd_ddd9
                        re_x3i = x3i
            if re_x3i != []:
                re_areas[yyy] = re_x3i
        if distance(re_areas[0][0],re_areas[0][1],re_areas[1][0],re_areas[1][1]) > distance(re_segment[0][0],re_segment[0][1],re_segment[-1][0],re_segment[-1][1]):
            re_areas = point_group

        print("before door: ", point_group," after door: ", re_areas)
        if point_group[0][0] == re_areas[0][0] and point_group[0][1] == re_areas[0][1] and point_group[1][0] == re_areas[1][0] and point_group[1][1] == re_areas[1][1]:
            print("&same&")
        # print("before door: ", point_group," after door: ", re_areas)

        return re_areas



    def check_re(self):
        # print('*****check*****')
        flag = True
        json_name = os.listdir(self.segmentation_data_path)
        tmp_existing_ids = [map_i.split('.')[0] for map_i in json_name]
        if len(self.unsaved_map_ids) == len(self.unsaved_map_data):
            if len(self.target_map_ids) == ((len(self.saved_map_ids)+len(self.skip_ids))+len(self.unsaved_map_ids)):
                for i in self.target_map_ids:
                    if i not in self.saved_map_ids:
                        if i not in self.unsaved_map_ids:
                            if i not in self.skip_ids:
                                print("[] target_map_ids data error1")
                                print("[] skip_ids: ", self.skip_ids, "name: ",i)
                                flag = False
                                break
                    else:
                        if i not in tmp_existing_ids:
                            print("[] "+i+" saved error")
                            flag = False
                            break
            else:
                print("[] target_map_ids len error")
                flag = False
        else:
            print("[] unsaved data")
            flag = False
        return flag
        
    def read_file(self, map_id):
        json_data = {}
        with open(self.original_map_path + '/' + map_id + '.json') as json_file:
            json_data = json.load(json_file)
        
        if json_data == {}:
            return_json_data = {}
        else:
            return_json_data = {map_id:{'verts':json_data['verts'], 'areas':[], 'door':[]}}
        return return_json_data


    def draw_map_room(self, json_data, id):
        # print("Processing ", file_name)

        # Draw the contour
        verts = (np.array(json_data[id]['verts']) * self.meter2pixel).astype(np.int)
        x_max, x_min, y_max, y_min = np.max(verts[:, 0]), np.min(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 1])  #vert存的是像素坐标（第一个元素是x，第二个元素是y）
        cnt_map = np.zeros((y_max - y_min + self.border_pad * 2+1,
                            x_max - x_min + self.border_pad * 2+1)) #cnt_map是图像矩阵坐标，第一个元素是图像的行数对应像素的y，第二个元素是图像的列数对应像素的x
        verts[:, 0] = verts[:, 0] - x_min + self.border_pad
        verts[:, 1] = verts[:, 1] - y_min + self.border_pad
        cv2.drawContours(cnt_map, [verts], 0, 255, -1)
        self.this_white_verts = verts
        return cnt_map, {"x_max":x_max, "x_min":x_min, "y_max":y_max, "y_min":y_min}


    # def point_transformer_json2image(self, points, x_min, y_min, border_pad):
    #     re = []
    #     for group in points:
    #         re_a = []
    #         for point in group:
    #             x = int(point[0] * self.meter2pixel - x_min + border_pad)
    #             y = int(point[1] * self.meter2pixel - y_min + border_pad)
    #             re_a.append([x,y])
    #         re.append(re_a)
    #     return re
    

    def point_transformer_json2image(self, points, x_min, y_min, border_pad):
        re = []
        for group in points:
            re_a = []
            for point in group:
                x = int(int(round(point[0] * self.meter2pixel)) - x_min + border_pad)
                y = int(int(round(point[1] * self.meter2pixel)) - y_min + border_pad)
                re_a.append([x,y])
            re.append(re_a)
        return re

    #备注：
    #int(point[0] * self.meter2pixel - x_min + border_pad) = int(int(round(point[0] * self.meter2pixel)) - x_min + border_pad)
    #后者才应该是根据point_transformer_image2json逆向推理得到的


    def point_transformer_image2json(self, points, x_min, y_min, border_pad):
        re = []
        for group in points:
            re_a = []
            for point in group:
                x = int(point[0] + x_min - border_pad)/float(self.meter2pixel)
                y = int(point[1] + y_min - border_pad)/float(self.meter2pixel)
                re_a.append([x,y])
            re.append(re_a)
        return re


    def load_image(self):
        print("\n\033[31m### Load Image \033[0m")
        self.text.insert(tk.END,'\n\n')
        self.text.see(tk.END)
        if self.map_index < len(self.target_map_ids):
            try:
                map_id = self.target_map_ids[self.map_index]
                print("\033[31m---{}---\033[0m".format(map_id))
                self.text.insert(tk.END,'\n ---{}---'.format(map_id))
                self.text.see(tk.END)
                json_data = self.read_file(map_id)
                map_matrix, map_bbox = self.draw_map_room(json_data, map_id) #map_matrix是图像矩阵(h,w)，map_bbox是像素坐标的坐标范围(x,y);x=w,y=h.
                self.flag_load = True
                if json_data == {}:
                    self.flag_load = False
            except:
                # print("read image file ")
                self.flag_load = False
            
            if self.flag_load == False:
                print('Fail to load!')
                self.text.insert(tk.END,'\n Fail to load!')
                self.text.see(tk.END)
                tk.messagebox.showinfo(title='error', message='Load image error!')
            else:
                height, width = map_matrix.shape[0], map_matrix.shape[1]
                if (height<=self.out_height) and (width <= self.out_width):
                    self.this_map_id = map_id
                    self.this_json_data = json_data
                    self.this_map = map_matrix #图像矩阵(h,w)
                    
                    self.this_image = Image.fromarray(self.this_map)
                    self.this_black_contour = self.search_black_conotur_pixel(self.this_map)#像素坐标(x,y)

                    self.this_black_verts, self.this_black_verts_re_vv_transition = self.search_black_verts(self.this_white_verts, self.this_map)#像素坐标(x,y), 首尾不重叠
                    self.this_box = map_bbox #像素坐标(x,y)
                    # self.display_save_points(self.this_black_contour) #测试时保存边界用
                    # print('self.this_black_contour: ', self.this_black_contour, len(self.this_black_contour))
                    plt.cla()
                    self.fig1.imshow(self.this_image)
                    self.canvs.draw()
                    print('Sucess to load!')
                    self.text.insert(tk.END,'\nSucess to load!')
                    self.text.see(tk.END)
                    print("[image info]: black_contour: {};black_verts: {}; black_verts_mini: {}".format(len(self.this_black_contour), len(self.this_black_verts), len(self.this_black_verts_re_vv_transition)))
                    target_map = self.simple_image_generation()#把加载的self.this_map原始图像转化成512x512的仿珊格地图彩色图target_map(障碍物只占一层边界，其他是未知和空闲区域，是没有标签的原始地图)
                    self.this_dataset_map = target_map
                    self.this_dataset_map_name = self.this_map_id + "_simple_v1.png"
                    self.map_index += 1
                    #test black verts
                    # self.show_black_verts()
                else:
                    self.flag_load = False
            if self.flag_load == False:
                self.clear_this()
            return False,False
        elif self.map_index == len(self.target_map_ids):
            print("\033[32m End loading\033[0m")
            if len(self.target_map_ids) > 0:
                if self.check_re():
                    print('All annotations completed')
                    self.text.insert(tk.END,'\nAll annotations completed')
                    self.text.see(tk.END)
                    tk.messagebox.showinfo(title='!!!!', message='All map annotations completed.')
                    return True, 1
                else:
                    print('annotations completed but ID CHECK failed')
                    self.text.insert(tk.END,'\nannotations completed but ID CHECK failed')
                    self.text.see(tk.END)
                    tk.messagebox.showwarning(title='Error', message='ID CHECK failed.')
                    return True, 2
            else:
                print('No need to annotate ')
                self.text.insert(tk.END,'\nNo need to annotate ')
                self.text.see(tk.END)
                tk.messagebox.showinfo(title='!!!!', message='No need to annotate.')
                return True, 3

        else:
            print('map_index > target_map_ids')
            self.text.insert(tk.END,'\nmap_index > target_map_ids')
            self.text.see(tk.END)
            tk.messagebox.showwarning(title='Error', message='map_index > target_map_ids')
            return True, 2


    def skip_error_data(self):
        print("\033[33m#Skip *****\033[0m")
        re = skip_attribute()
        
        if re.result == -1:
            print("skip error")
            self.text.insert(tk.END,'\n [skip error]')
            self.text.see(tk.END)
            tk.messagebox.showinfo(title='Error', message='skip type error!')
            print("unsaved_map_ids: ", len(self.unsaved_map_ids),"; unsaved_map_data: ", len(self.unsaved_map_data), "unsaved_dataset_image", len(self.unsaved_dataset_image))
            print("saved_map_ids: ", len(self.saved_map_ids))
            print("skip_ids: ", len(self.skip_ids))
            print("target_ids: ", len(self.target_map_ids))
            print("\033[33m ***** Skip End\033[0m")
        elif re.result == 0:
            print("to next")
            self.text.insert(tk.END,'\n [skip to next]')
            self.text.see(tk.END)
            self.skip_ids.append(self.this_map_id)
            self.clear_this()
            print("unsaved_map_ids: ", len(self.unsaved_map_ids),"; unsaved_map_data: ", len(self.unsaved_map_data), "unsaved_dataset_image", len(self.unsaved_dataset_image))
            print("saved_map_ids: ", len(self.saved_map_ids))
            print("skip_ids: ", len(self.skip_ids))
            print("target_ids: ", len(self.target_map_ids))
            print("map_index: ",self.map_index)
            print("\033[33m ***** Skip End\033[0m")
            if len(self.unsaved_map_ids) == len(self.unsaved_map_data):
                end_flag, end_type = self.load_image()
                if end_flag == True:
                    if end_type == 1:
                        if self.save_segmentation_data():
                            # self.test()
                            plt.close('all')
                            print("END")
                            self.root.destroy()
                    else:
                        tk.messagebox.showinfo(title='error', message='saving error!')
                        print("[] unsaved_map_ids: ", len(self.unsaved_map_ids),"; unsaved_map_data: ", len(self.unsaved_map_data), "unsaved_dataset_image", len(self.unsaved_dataset_image))
                        print("[] saved_map_ids: ", len(self.saved_map_ids))
                        print("[] skip_ids: ", len(self.skip_ids))
                        print("[] map_index: ",self.map_index)
                        print("[] target_ids: ", len(self.target_map_ids))
            else:
                tk.messagebox.showinfo(title='error', message='unsaved map data error !')
                print("[] unsaved_map_ids: ", len(self.unsaved_map_ids),"; unsaved_map_data: ", len(self.unsaved_map_data), "unsaved_dataset_image", len(self.unsaved_dataset_image))
                print("[] saved_map_ids: ", len(self.saved_map_ids))
                print("[] skip_ids: ", len(self.skip_ids))
                print("[] map_index: ",self.map_index)
                print("[] target_ids: ", len(self.target_map_ids))
        else:
            print("reload this")
            self.text.insert(tk.END,'\n [reload this one]')
            self.text.see(tk.END)
            print("unsaved_map_ids: ", len(self.unsaved_map_ids),"; unsaved_map_data: ", len(self.unsaved_map_data), "unsaved_dataset_image", len(self.unsaved_dataset_image))
            print("saved_map_ids: ", len(self.saved_map_ids))
            print("skip_ids: ", len(self.skip_ids))
            print("target_ids: ", len(self.target_map_ids))
            self.clear_this()
            self.map_index = self.map_index -1
            print("map_index: ",self.map_index)
            print("\033[33m ***** Skip End\033[0m")
            if len(self.unsaved_map_ids) == len(self.unsaved_map_data):
                end_flag, end_type = self.load_image()
                if end_flag == True:
                    if end_type == 1:
                        if self.save_segmentation_data():
                            # self.test()
                            plt.close('all')
                            print("END")
                            self.root.destroy()
                    else:
                        print("[] unsaved_map_ids: ", len(self.unsaved_map_ids),"; unsaved_map_data: ", len(self.unsaved_map_data), "unsaved_dataset_image", len(self.unsaved_dataset_image))
                        print("[] saved_map_ids: ", len(self.saved_map_ids))
                        print("[] skip_ids: ", len(self.skip_ids))
                        print("[] map_index: ",self.map_index)
                        print("[] target_ids: ", len(self.target_map_ids))
                        tk.messagebox.showinfo(title='error', message='saving error!')
            else:
                tk.messagebox.showinfo(title='error', message='unsaved map data error !')
                print("[] unsaved_map_ids: ", len(self.unsaved_map_ids),"; unsaved_map_data: ", len(self.unsaved_map_data), "unsaved_dataset_image", len(self.unsaved_dataset_image))
                print("[] saved_map_ids: ", len(self.saved_map_ids))
                print("[] skip_ids: ", len(self.skip_ids))
                print("[] target_ids: ", len(self.target_map_ids))
                print("[] map_index: ",self.map_index)

    def test(self):
        print("json data verification")
        out = True
        num = 0
        for id in self.test_list:
            if id in self.saved_map_ids:
                with open(self.segmentation_data_path + '/' + id + '.json') as json_file:
                    data = json.load(json_file)
                verts = (np.array(data['verts']) * self.meter2pixel).astype(np.int)
                x_max, x_min, y_max, y_min = np.max(verts[:, 0]), np.min(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 1])
                data2 = self.point_transformer_json2image(data['areas'], x_min=x_min, y_min=y_min,border_pad= self.border_pad)
            else:
                out = False
                print("can not find data for test id: ",id)
                break
            if len(self.test_list[id]) == len(data2):
                for i in range(len(self.test_list[id])):
                    for i_j in range(len(self.test_list[id][i])):
                        if (self.test_list[id][i][i_j][0] == data2[i][i_j][0]) and (self.test_list[id][i][i_j][1] == data2[i][i_j][1]):
                            num += 1
                        else:
                            out = False
                            print(id,'  X X X X')
                            break
            else:
                out = False
            if out == False:
                break
            # if out == True:
                # print(id,"{} points".format(num))
        if out == True:
            print("verification sucess")
            tk.messagebox.showinfo(title='result', message='verification success')
        else:
            print("verification fail")
            tk.messagebox.showinfo(title='result', message='verification fail')



    def test2(self, pth1, pth2, da_type, list_type):
        print("json data verification")
        out = True
        num = 0
        for id in list_type:
            if id in self.saved_map_ids:
                with open(pth1 + '/' + id + '.json') as json_file:
                    data = json.load(json_file)
                with open(pth2 + '/' + id + '.json') as json_file2:
                    data_1 = json.load(json_file2)
                verts = (np.array(data['verts']) * self.meter2pixel).astype(np.int)
                x_max, x_min, y_max, y_min = np.max(verts[:, 0]), np.min(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 1])
                data2 = self.point_transformer_json2image(data_1[da_type], x_min=x_min, y_min=y_min,border_pad= self.border_pad)
            else:
                out = False
                print("can not find data for test id: ",id)
                break
            if len(list_type[id]) == len(data2):
                for i in range(len(list_type[id])):
                    for i_j in range(len(list_type[id][i])):
                        if (list_type[id][i][i_j][0] == data2[i][i_j][0]) and (list_type[id][i][i_j][1] == data2[i][i_j][1]):
                            num += 1
                        else:
                            out = False
                            print(id,'  X X X X')
                            break
            else:
                out = False
            if out == False:
                break
            if out == True:
                print(id,"{} points".format(num))
        if out == True:
            print("verification sucess")
            tk.messagebox.showinfo(title='result', message='verification success')
        else:
            print("verification fail")
            tk.messagebox.showinfo(title='result', message='verification fail')

    def test3(self, pth1, pth2):
        print("json data verification")
        out = True
        num = 0
        file_name1 = os.listdir(pth1)
        rfiles1 = [map_i.split('.')[0] for map_i in file_name1]
        file_name2 = os.listdir(pth2)
        rfiles2 = [map_i.split('.')[0] for map_i in file_name2]
        for id in rfiles1:
            if id in rfiles2:
                with open(pth1 + '/' + id + '.json') as json_file:
                    data = json.load(json_file)
                with open(pth2 + '/' + id + '.json') as json_file2:
                    data_1 = json.load(json_file2)
            else:
                out = False
                print("can not find data for test id: ",id)
                break
            if len(data['areas']) == len(data_1['areas']):
                for i in range(len(data['areas'])):
                    for i_j in range(len(data['areas'])):
                        if (data['areas'][i][i_j][0] == data_1['areas'][i][i_j][0]) and (data['areas'][i][i_j][1] == data_1['areas'][i][i_j][1]):
                            num += 1
                        else:
                            out = False
                            print(id,'  X X X X')
                            break
            else:
                out = False
            if len(data['door']) == len(data_1['door']):
                for i in range(len(data['door'])):
                    for i_j in range(len(data['door'])):
                        if (data['door'][i][i_j][0] == data_1['door'][i][i_j][0]) and (data['door'][i][i_j][1] == data_1['door'][i][i_j][1]):
                            num += 1
                        else:
                            out = False
                            print(id,'  X X X X')
                            break
            else:
                out = False
            if out == False:
                break
            if out == True:
                print(id,"{} points".format(num))
        if out == True:
            print("verification sucess")
            tk.messagebox.showinfo(title='result', message='verification success')
        else:
            print("verification fail")
            tk.messagebox.showinfo(title='result', message='verification fail')



    def render(self):
        pass
    def last_map(self):
        pass

    def map_info(self):
        pass

    def delete_area(self):
        if len(self.this_json_data[self.this_map_id]['door']) > 0:
            print("\033[33m#Delete Door *****\033[0m")
            print("Existing area / door: ", len(self.this_json_data[self.this_map_id]['areas']),"/", len(self.this_json_data[self.this_map_id]['door']))
            self.this_json_data[self.this_map_id]['door'].pop()
            print("New area / door: ", len(self.this_json_data[self.this_map_id]['areas']),"/", len(self.this_json_data[self.this_map_id]['door']))
            self.text.insert(tk.END,'\n Door number --> {}'.format(len(self.this_json_data[self.this_map_id]['door'])))
            self.text.see(tk.END)
        else:
            print("\033[33m#Delete Area *****\033[0m")
            print("Existing area / door: ", len(self.this_json_data[self.this_map_id]['areas']),"/", len(self.this_json_data[self.this_map_id]['door']))
            if len(self.this_json_data[self.this_map_id]['areas']) > 0:
                self.this_json_data[self.this_map_id]['areas'].pop()
                print("New area / door: ", len(self.this_json_data[self.this_map_id]['areas']),"/", len(self.this_json_data[self.this_map_id]['door']))
                self.text.insert(tk.END,'\n room number --> {}'.format(len(self.this_json_data[self.this_map_id]['areas'])))
                self.text.see(tk.END)
            else:
                tk.messagebox.showwarning(title='Warning',message='No more area!')
                print("[] No area to delete! ")
                print("The area / door: ", len(self.this_json_data[self.this_map_id]['areas']),"/", len(self.this_json_data[self.this_map_id]['door']))
            # if len(self.this_json_data[self.this_map_id]['door']) > len(self.this_json_data[self.this_map_id]['areas']):
            #     self.this_json_data[self.this_map_id]['door'].pop()
            # self.this_flag_consistent_0=False
            # self.this_flag_consistent_1=False
            if len(self.this_json_data[self.this_map_id]['areas']) == 0:
                self.flag_area = False
            # print("delet result area-door: ", self.this_json_data[self.this_map_id]['areas'], self.this_json_data[self.this_map_id]['door'])
        self.draw_target2()
        print("\033[33m ***** Delete End\033[0m")


    def save_segmentation_data(self):
        print("\033[34m### Save \033[0m")
        self.text.insert(tk.END,'\nsaving......')
        self.text.see(tk.END)
        flag_2 = True
        if len(self.unsaved_map_ids) == 0 and len(self.unsaved_dataset_image) == 0:
            print("No need to save anything")
            self.text.insert(tk.END,"\nNo need to save anything")
            print("\033[34m### Save End\033[0m")
            return True
            
        print("[before 1] saved_map_ids: ",len(self.saved_map_ids),"; unsaved_map_ids: ",len(self.unsaved_map_ids))
        success_list = []
        save_num = len(self.unsaved_map_ids)
        while len(self.unsaved_map_ids) > 0:
            flag_1 = False
            # print(self.unsaved_map_data)
            data = self.unsaved_map_data[self.unsaved_map_ids[0]]
            try:
                with open(self.segmentation_data_path + '/' + self.unsaved_map_ids[0] + '.json', 'w') as f:
                    json.dump(data, f)
                flag_1 = True
                if self.record_data_flag == False:
                    success_list.append(self.unsaved_map_ids[0])
            except:
                flag_1 = False

            if self.record_data_flag == True and flag_1 == True:
                flag_1 = False
                # print(self.unsaved_map_data)
                data_record_ = self.unsaved_record_data[self.unsaved_map_ids[0]]
                try:
                    with open(self.record_data_path + '/' + self.unsaved_map_ids[0] + '.json', 'w') as f:
                        json.dump(data_record_, f)
                    flag_1 = True
                    success_list.append(self.unsaved_map_ids[0])
                except:
                    flag_1 = False

            if flag_1 == True:
                # print("\033[36m--segment success\033[0m")
                # self.text.insert(tk.END,"\n-<segment success")
                # self.text.see(tk.END)
                self.saved_map_ids.append(self.unsaved_map_ids[0])
                del self.unsaved_map_data[self.unsaved_map_ids[0]]
                if self.record_data_flag == True:
                    del self.unsaved_record_data[self.unsaved_map_ids[0]]
                self.unsaved_map_ids.pop(0)

                # print("save: unsaved_map_ids, unsaved_map_data, saved_map_ids", self.unsaved_map_ids, self.unsaved_map_data, self.saved_map_ids)
                # print("len(unsaved_map_data), len(unsaved_map_ids), len(saved_map_ids): ", len(self.unsaved_map_data), len(self.unsaved_map_ids), len(self.saved_map_ids))
            else: 
                print("\033[36m--segment fail\033[0m")
                self.text.insert(tk.END,"\n-<segment fail")
                self.text.see(tk.END)
                # tk.messagebox.showinfo(title='error', message='fail to save segmentation'+self.unsaved_map_ids[0])
                flag_2 = False
                break
        if len(success_list) == save_num and flag_2 == True:
            print("\033[36m--all {} segmentations success\033[0m".format(save_num))
            self.text.insert(tk.END,"\n--<all {} segmentations success".format(save_num))
            self.text.see(tk.END)
            exsting_name_lines = []
            new_success_list = []
            if os.path.exists(self.segmentation_data_list_path + '/' + "json_map_list.txt"):
                with open(self.segmentation_data_list_path + '/' + "json_map_list.txt", 'r') as ff_l:
                    exsting_name_lines = ff_l.readlines()
            for i_check in success_list:
                if i_check+"\n" not in exsting_name_lines:
                    new_success_list.append(i_check)
                else:
                    print("\033[36m--warning: {} existed in map list\033[0m".format(i_check))
            # print("exsting_name_lines   ",exsting_name_lines) #test
            with open(self.segmentation_data_list_path + '/' + "json_map_list.txt", 'a') as ff_l:
                for ff_l_row in range(len(new_success_list)):
                        ff_l.write(new_success_list[ff_l_row])
                        ff_l.write("\n")
        else:
            print("\033[36m--only {} / {} maps saved\033[0m".format(len(success_list),save_num))
            self.text.insert(tk.END,"\n--<only {} / {} maps saved".format(len(success_list),save_num))
            self.text.see(tk.END)
            exsting_name_lines = []
            new_success_list = []
            if os.path.exists(self.segmentation_data_list_path + '/' + "json_map_list.txt"):
                with open(self.segmentation_data_list_path + '/' + "json_map_list.txt", 'r') as ff_l:
                    exsting_name_lines = ff_l.readlines()
            for i_check in success_list:
                if i_check+"\n" not in exsting_name_lines:
                    new_success_list.append(i_check)
                else:
                    print("\033[36m--warning: {} existed in map list\033[0m".format(i_check))
            with open(self.segmentation_data_list_path + '/' + "json_map_list.txt", 'a') as ff_l:
                for ff_l_row in range(len(new_success_list)):
                        ff_l.write(new_success_list[ff_l_row])
                        ff_l.write("\n")
        print("[after 1] saved_map_ids: ",len(self.saved_map_ids),"; unsaved_map_ids: ",len(self.unsaved_map_ids))
        #以上部分是保存用于后续数据集扩增的边界数据库。以下是保存的简易数据集，简易数据集这部分还需后期和数据集扩增器的相统一
        if flag_2 == True:
            if self.simple_dataset_generate_flag == True:
                # self.text.insert(tk.END,'\nsimple dataset saving: ')
                # self.text.see(tk.END)
                # self.text.insert(tk.END,'\ndataset generating:  ')
                # self.text.see(tk.END)
                print("[before 2]:  unsaved_dataset_image: ",len(self.unsaved_dataset_image))
                delte_name = []
                flag_3 = True
                num_saved_dataset = len(self.unsaved_dataset_image)
                for data_name in self.unsaved_dataset_image:
                    try:
                        # cv2.imwrite(self.dataset_simple_image_path+'/'+data_name, self.unsaved_dataset_image[data_name])
                        pass
                        delte_name.append(data_name)
                    except:
                        flag_3 = False
                        flag_2 = False
                        break
                if flag_3 == True:
                    if len(self.unsaved_dataset_image) > 0:
                        flag_4 = True
                        try:
                            # self.save_simple_dataset_json()
                            pass
                        except:             
                            flag_4 = False
                            flag_2 = False
                        if flag_4 == True:
                            for i_d in delte_name:
                                del self.unsaved_dataset_image[i_d]
                            if self.unsaved_dataset_image != {}:
                                tk.messagebox.showinfo(title='error', message='fail to save image')
                                flag_2 = False
                        else:
                            tk.messagebox.showinfo(title='error', message='fail to save anno')
                    else:
                        print("No need to save dataset")
                else: 
                    tk.messagebox.showinfo(title='error', message='fail to save image---' + data_name)
                    delte_name = []
                # print("[after 2]:  unsaved_dataset_image: ",len(self.unsaved_dataset_image))
                if flag_2 == True:
                    print("\033[36m--all {} dataset success\033[0m".format(len(delte_name)))
                    self.text.insert(tk.END,"\n-<all {} dataset success".format(len(delte_name)))
                    self.text.see(tk.END)
                else:
                    print("\033[36m--Failure: only {} / {} dataset saved\033[0m".format(len(delte_name), len(self.unsaved_dataset_image)))
                    self.text.insert(tk.END,"\n-<Failure: only {} / {} dataset saved".format(len(delte_name), len(self.unsaved_dataset_image)))
                    self.text.see(tk.END)
                print("[after 2]:  unsaved_dataset_image: ",len(self.unsaved_dataset_image))
        else:
            tk.messagebox.showinfo(title='error', message='Failed to save segmentation data')
        print("\033[34m### Save End\033[0m")
        return flag_2

    def training_data_generation(self, data):
        self.dataset_image_count = self.dataset_image_count + 1
        for contours in data:
            polys = Polygon(contours)
            area = int(round(polys.area))
            bounds = polys.bounds 
            x = int(round(bounds[0]))
            y = int(round(bounds[1]))
            w = int(round(bounds[2]-bounds[0]+1))
            h = int(round(bounds[3]-bounds[1]+1))
            self.dataset_annotations_count = self.dataset_annotations_count + 1
            contours = [[c for b in contours for c in b ]]
            anno = {'segmentation': contours, 'area': area,
                    'image_id': self.dataset_image_count, 'bbox': [x, y, w, h],
                    'iscrowd': 0, 'category_id': 1, 'id': self.dataset_annotations_count}
            self.dataset_annotations.append(anno)
        
        info = {'file_name': self.this_dataset_map_name,
                'height': 512, 'width': 512, 'id': self.dataset_image_count}
        self.dataset_mages_info.append(info)



    def save_simple_dataset_json(self):
        categories = [
            {'supercategory': 'none', 'id': 1, 'name': 'room'}
        ]
        palce_instance = {'images': self.dataset_mages_info, 'annotations': self.dataset_annotations, 'categories': categories}
        # print(self.dataset_mages_info)
        # print(self.dataset_annotations)

        # os.system('mkdir -p {}'.format(path))
        with open(self.dataset_simple_anno_path, 'w') as f:
            json.dump(palce_instance, f)





    def clear_this(self):
        self.flag_area = False

        self.this_map_id = ''
        self.this_black_contour = []
        self.this_json_data = {}
        self.this_map = np.zeros((500,500))
        self.this_image = Image.fromarray(self.this_map)
        self.this_box = {}
        self.this_dataset_map = np.full((self.out_height,self.out_width,3),220, dtype=np.uint8)
        self.this_dataset_contours = []
        self.this_dataset_map_name = ''
        self.this_offset = {'offset_height':-1,'offset_width':-1}
        self.this_flag_consistent_0=False
        self.this_flag_consistent_1=False
        self.this_record_data = {}
        
    

    def simple_data_set(self):
        pass


    def confirm(self):
        print("\033[33m#Confirm *****\033[0m")
        if len(self.this_json_data[self.this_map_id]['areas']) == 0:
            self.flag_area = 0
        if self.flag_area == True and self.flag_load == True:
            self.dataset_display_flag = True
            if self.dataset_display_flag == True:
                # print("before simple:", type(self.this_black_contour[0][0]))
                # print("before simple2:", type(self.this_json_data[self.this_map_id]['areas'][0][0][0]))
                if self.record_data_flag == True:
                    tmp_for_test_list2 = copy.deepcopy(self.this_json_data[self.this_map_id]['areas'])
                    tran_areas = self.point_transformer_image2json(self.this_json_data[self.this_map_id]['areas'],x_min=int(self.this_box["x_min"]),y_min=int(self.this_box["y_min"]), border_pad=self.border_pad)
                    self.this_record_data = {self.this_map_id:{'areas':tran_areas, 'door':[]}}
                    
                re_areas = self.simple_anno_generation_v2() #re_areas对之前记录的区域点沿着边界间隔采样并扩增，仍然是图片像素点并且首尾重复
                # print("after simple:", type(re_areas[0][0][0]))
                # print("re_areas: ",re_areas)
                self.this_dataset_contours = []
                if (self.this_offset['offset_height'] != -1) and (self.this_offset['offset_width'] != -1):
                    for i in range(len(re_areas)):
                        temp_data = []
                        for j in range(len(re_areas[i])):
                            temp_data.append([int(re_areas[i][j][0] + self.this_offset['offset_width']), int(re_areas[i][j][1] + self.this_offset['offset_height'])])
                        self.this_dataset_contours.append(temp_data)
                else:
                    next_flag == False
                    tk.messagebox.showinfo(title='warning', message='offset error!!!')
                # self.this_dataset_contours = re_areas

                displayer = picture_display(self.this_dataset_map,self.this_dataset_contours)
                next_flag = displayer.get_flag()
                # print("nexr_flag: ",next_flag)
            else:
                next_flag == True
            #这里有个缺陷，下面的边界数据实际上用的是re_areas（经过沿着边界间隔采样并扩增后的边界点），所以self.dataset_display_flag必须取true才能保证以下的语法和逻辑
            if next_flag == True:
                if len(self.unsaved_map_ids) == len(self.unsaved_map_data):
                    if len(self.unsaved_map_ids) < 10:
                        self.test_list.update({self.this_map_id:re_areas})
                        self.test_list3.update({self.this_map_id:self.this_json_data[self.this_map_id]['door']})
                        # self.this_json_data[self.this_map_id]['areas'] = self.point_transformer_image2json(self.this_json_data[self.this_map_id]['areas'],x_min=int(self.this_box["x_min"]),y_min=int(self.this_box["y_min"]), border_pad=self.border_pad)
                        self.this_json_data[self.this_map_id]['areas'] = self.point_transformer_image2json(re_areas,x_min=int(self.this_box["x_min"]),y_min=int(self.this_box["y_min"]), border_pad=self.border_pad)#临时改动
                        self.this_json_data[self.this_map_id]['door'] = self.point_transformer_image2json(self.this_json_data[self.this_map_id]['door'],x_min=int(self.this_box["x_min"]),y_min=int(self.this_box["y_min"]), border_pad=self.border_pad)
                        self.unsaved_map_data.update(self.this_json_data)
                        self.unsaved_map_ids.append(self.this_map_id)
                        self.unsaved_dataset_image.update({self.this_dataset_map_name:self.this_dataset_map})
                        if self.record_data_flag == True:
                            self.this_record_data[self.this_map_id]['door'] = self.this_json_data[self.this_map_id]['door']
                            self.unsaved_record_data.update(self.this_record_data)
                            self.test_list2.update({self.this_map_id:tmp_for_test_list2})
                        self.training_data_generation(self.this_dataset_contours)
                        print("unsaved_map_ids: ", len(self.unsaved_map_ids),"; unsaved_map_data: ", len(self.unsaved_map_data), "unsaved_dataset_image", len(self.unsaved_dataset_image), "unsaved_record_data", len(self.unsaved_record_data))
                        print("saved_map_ids: ", len(self.saved_map_ids))
                        print("skip_ids: ", len(self.skip_ids))
                        print("target_ids: ", len(self.target_map_ids))
                        print("map_index: ",self.map_index)
                        print("confirm success")
                        self.text.insert(tk.END,'\n [unsaved] map: {}; dataset: {}'.format(len(self.unsaved_map_ids), len(self.unsaved_dataset_image)))
                        self.text.insert(tk.END,'\n [saved] map: {}'.format(len(self.saved_map_ids)))
                        self.text.insert(tk.END,'\n confirm success')
                        self.text.see(tk.END)
                    else:
                        self.save_segmentation_data() 
                        # （1）将self.unsaved_map_data对应的self.this_json_data[self.this_map_id]保存到map_id.json，作为后续扩增数据集的边界数据库，（2）之后利用self.this_dataset_contours生成可用的简单数据集
                        # self.this_json_data[self.this_map_id]里面包含'areas'（其实就是re_areas）和'verts'两个元素
                        # 'areas'和'verts'都是图片像素坐标(x,y)都与图像矩阵的索引是反着的,x对应矩阵的列，而y对应矩阵的行


                        self.test_list.update({self.this_map_id:re_areas})
                        self.test_list3.update({self.this_map_id:self.this_json_data[self.this_map_id]['door']})
                        # self.this_json_data[self.this_map_id]['areas'] = self.point_transformer_image2json(self.this_json_data[self.this_map_id]['areas'],x_min=int(self.this_box["x_min"]),y_min=int(self.this_box["y_min"]), border_pad=self.border_pad)
                        self.this_json_data[self.this_map_id]['areas'] = self.point_transformer_image2json(re_areas,x_min=int(self.this_box["x_min"]),y_min=int(self.this_box["y_min"]), border_pad=self.border_pad)
                        self.this_json_data[self.this_map_id]['door'] = self.point_transformer_image2json(self.this_json_data[self.this_map_id]['door'],x_min=int(self.this_box["x_min"]),y_min=int(self.this_box["y_min"]), border_pad=self.border_pad)
                        self.unsaved_map_data.update(self.this_json_data)
                        self.unsaved_map_ids.append(self.this_map_id)
                        self.unsaved_dataset_image.update({self.this_dataset_map_name:self.this_dataset_map})
                        if self.record_data_flag == True:
                            self.this_record_data[self.this_map_id]['door'] = self.this_json_data[self.this_map_id]['door']
                            self.unsaved_record_data.update(self.this_record_data)
                            self.test_list2.update({self.this_map_id:tmp_for_test_list2})
                        self.training_data_generation(self.this_dataset_contours)
                        # print("unsaved_map_ids: ", len(self.unsaved_map_data))
                        # print("unsaved_map_data: ", len(self.unsaved_map_ids))
                        print("unsaved_map_ids: ", len(self.unsaved_map_ids),"; unsaved_map_data: ", len(self.unsaved_map_data), "unsaved_dataset_image", len(self.unsaved_dataset_image),"unsaved_record_data", len(self.unsaved_record_data))
                        print("saved_map_ids: ", len(self.saved_map_ids))
                        print("skip_ids: ", len(self.skip_ids))
                        print("target_ids: ", len(self.target_map_ids))
                        print("map_index: ",self.map_index)
                        print("confirm success")
                        self.text.insert(tk.END,'\n [unsaved] map: {}; dataset: {}'.format(len(self.unsaved_map_ids), len(self.unsaved_dataset_image)))
                        self.text.insert(tk.END,'\n [saved] map: {}'.format(len(self.saved_map_ids)))
                        self.text.insert(tk.END,'\n confirm success')
                        self.text.see(tk.END)
                    print("\033[33m ***** Confirm End\033[0m")
                    self.clear_this()
                    end_flag, end_type = self.load_image()
                    if end_flag == True:
                        end_saving_flag = True
                        if end_type == 1:
                            if self.save_segmentation_data():
                                # self.test()
                                # self.test2(self.segmentation_data_path, self.segmentation_data_path, 'door', self.test_list3)
                                # if self.record_data_flag == True:
                                #     self.test2(self.segmentation_data_path, self.record_data_path, 'areas', self.test_list2)
                                plt.close('all')
                                print("END")
                                self.root.destroy()
                            else:
                                end_saving_flag = False
                        if end_type != 1 or end_saving_flag == False:
                            print("[] end_type: ", end_type)
                            print("[] unsaved_map_ids: ", len(self.unsaved_map_ids),"; unsaved_map_data: ", len(self.unsaved_map_data), "unsaved_dataset_image", len(self.unsaved_dataset_image), "unsaved_record_data", len(self.unsaved_record_data))
                            print("[] saved_map_ids: ", len(self.saved_map_ids))
                            print("[] skip_ids: ", len(self.skip_ids))
                            print("[] map_index: ",self.map_index)
                            print("[] target_ids: ", len(self.target_map_ids))
                            print("saving error!")
                            tk.messagebox.showinfo(title='error', message='saving error!')
                else:
                    print("[] unsaved_map_ids: ", len(self.unsaved_map_ids),"; unsaved_map_data: ", len(self.unsaved_map_data), "unsaved_dataset_image", len(self.unsaved_dataset_image), "unsaved_record_data", len(self.unsaved_record_data))
                    print("[] saved_map_ids: ", len(self.saved_map_ids))
                    print("[] skip_ids: ", len(self.skip_ids))
                    print("[] map_index: ",self.map_index)
                    print("[] target_ids: ", len(self.target_map_ids))
                    tk.messagebox.showinfo(title='error', message='unsaved map data error !')
                    print("'unsaved map data error !'")
                    print("\033[33m ***** Confirm End\033[0m")
            else:
                print("[cannot confirm] data setting faid")
                print("[] unsaved_map_ids: ", len(self.unsaved_map_ids),"; unsaved_map_data: ", len(self.unsaved_map_data), "unsaved_dataset_image", len(self.unsaved_dataset_image), "unsaved_record_data", len(self.unsaved_record_data))
                print("[] saved_map_ids: ", len(self.saved_map_ids))
                print("[] skip_ids: ", len(self.skip_ids))
                print("[] map_index: ",self.map_index)
                print("[] target_ids: ", len(self.target_map_ids))
                self.text.insert(tk.END,'\n [cannot confirm] data setting faid')
                self.text.see(tk.END)
                tk.messagebox.showinfo(title='warning', message='[cannot confirm] data setting faid')
                print("\033[33m ***** Confirm End\033[0m")
        else:
            print("[cannot confirm] incomplete data")
            print("[] unsaved_map_ids: ", len(self.unsaved_map_ids),"; unsaved_map_data: ", len(self.unsaved_map_data), "unsaved_dataset_image", len(self.unsaved_dataset_image),"unsaved_record_data", len(self.unsaved_record_data))
            print("[] saved_map_ids: ", len(self.saved_map_ids))
            print("[] skip_ids: ", len(self.skip_ids))
            print("[] map_index: ",self.map_index)
            print("[] target_ids: ", len(self.target_map_ids))
            self.text.insert(tk.END,'\n [cannot confirm] incomplete data')
            self.text.see(tk.END)
            tk.messagebox.showinfo(title='warning', message='incomplete data: cannot confirm!')
            print("\033[33m ***** Confirm End\033[0m")


    def exit(self):
        if self.save_segmentation_data():
            print("unsaved_map_ids: ", len(self.unsaved_map_ids),"; unsaved_map_data: ", len(self.unsaved_map_data), "unsaved_dataset_image", len(self.unsaved_dataset_image),"unsaved_record_data", len(self.unsaved_record_data))
            print("saved_map_ids: ", len(self.saved_map_ids))
            print("skip_ids: ", len(self.skip_ids))
            print("target_ids: ", len(self.target_map_ids))
            plt.close('all')
            print("END")
            self.root.destroy()
        else:
            print("Exist Error")
            tk.messagebox.showwarning(title='Error',message='Exist Error')



            
    def on_closing(self):
        plt.close('all')
        print("END")
        self.root.destroy()

    

    # def GeneralEquation(self, first_x,first_y,second_x,second_y):
    #     A = second_y-first_y
    #     B = first_x-second_x
    #     C = second_x*first_y-first_x*second_y
    #     if B != 0:
    #         k = -1 * A / B
    #         b = -1 * C / B

        
    #     A1 = first_y - second_y
    #     B2 = first_x-second_x
    #     K2 = A1/B2
    #     b2 = first_y - K2 * first_x
        
    #     return k, b



    def search_black_verts(self, verts_www, image):
        def k_gradient(x1, y1,x2,y2):
            a = x1 - x2
            b = y1 - y2
            k = math.atan2(b,a)
            if k < 0:
                k = k + 2*math.pi
            return k
        def distance_vv(x1,y1,x2,y2):
            return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
        

        verts_ooo = []
        auau = 0
        for i8 in range(len(verts_www)):
            if i8 > 0:
                if verts_www[i8][0] != verts_www[i8-1][0] or verts_www[i8][1] != verts_www[i8-1][1]:
                    verts_ooo.append(verts_www[i8])
                if verts_www[i8][0] == verts_www[i8-1][0] and verts_www[i8][1] == verts_www[i8-1][1]:
                    auau = auau + 1

            else:
                verts_ooo.append(verts_www[i8])

        if len(verts_ooo) < 3:
            print("1")
            return [],[]
        pp_012 = []
        for ooo in range(3):
            # print("verts_ooo:")
            # print(verts_ooo[ooo])
            temp_012 = []
            for i in range(-1,2):
                if (verts_ooo[ooo][1]+i) >=0 or (verts_ooo[ooo][1]+i) < len(image):
                    for j in range(-1,2):
                        if (verts_ooo[ooo][0]+j) >=0 or (verts_ooo[ooo][0]+j) < len(image[0]):
                            if image[verts_ooo[ooo][1]+i][verts_ooo[ooo][0]+j] == 0:
                                if i !=0 or j!=0:
                                    temp_012.append([verts_ooo[ooo][0]+j,verts_ooo[ooo][1]+i])
            # print("temp_012:")
            # print(temp_012)
            if temp_012 != []:
                pp_012.append(temp_012)
            else:
                print("2")
                return [],[]

        # print("pp_012:")
        # print(pp_012)


        min_k = [float('inf')]
        re_pp = []
        for pp_0 in pp_012[0]:
            for pp_1 in pp_012[1]:
                for pp_2 in pp_012[2]:
                    r1 = abs(k_gradient(pp_0[0], pp_0[1], pp_1[0], pp_1[1]) - k_gradient(verts_ooo[0][0], verts_ooo[0][1], verts_ooo[1][0], verts_ooo[1][1]))
                    if r1 > math.pi:
                        r1 = 2*math.pi - r1
                    r2 = abs(k_gradient(pp_1[0], pp_1[1], pp_2[0], pp_2[1]) - k_gradient(verts_ooo[1][0], verts_ooo[1][1], verts_ooo[2][0], verts_ooo[2][1]))
                    if r2 > math.pi:
                        r2 = 2*math.pi - r2
                    rrr =  r1+r2
                        
                    for hi in range(len(min_k)):
                        if min_k[hi] > rrr:
                            re_pp.insert(hi,[pp_0, pp_1, pp_2])
                            min_k.insert(hi,rrr)
                            min_k[-1] = min_k[0] + 0.018
                            for i in range(len(re_pp)):
                                if min_k[-1] <= min_k[i]:
                                    re_pp = re_pp[:i]
                                    min_k = min_k[:i]
                                    min_k.append(min_k[0] + 0.018)
                                    break
                            break
        # print("1. min_k: ", min_k)
        # print("1. re_pp: ", re_pp)
        if (len(re_pp) != (len(min_k)-1)):
            print("3")
            return [],[]

        max_lenl = -1
        re_pp1= []
        for hii in re_pp:
            qq = distance_vv(hii[0][0],hii[0][1], hii[1][0],hii[1][1]) + distance_vv(hii[1][0],hii[1][1], hii[2][0],hii[2][1])
            if qq > max_lenl:
                max_lenl = qq
                re_pp1 = hii
        re_pp = re_pp1
        # print("re_pp[3]", re_pp)


        if len(re_pp) != 3:
            print("4")
            return [],[]
        
        for iju in range(3,len(verts_ooo)):
            # min_k = float('inf')
            # # print(":::::: ")
            
            # re_pp_000 = []

            min_k = [float('inf')]
            re_pp_000 = []
            for i in range(-1,2):
                if (verts_ooo[iju][1]+i) >=0 or (verts_ooo[iju][1]+i) < len(image):
                    for j in range(-1,2):
                        if (verts_ooo[iju][0]+j) >=0 or (verts_ooo[iju][0]+j) < len(image[0]):
                            # print("+++: ", re_pp_000)
                            if image[verts_ooo[iju][1]+i][verts_ooo[iju][0]+j] == 0:
                                # print(re_pp_000)
                                if i !=0 or j!=0:
                                    rrr = abs(k_gradient(re_pp[-1][0], re_pp[-1][1],verts_ooo[iju][0]+j, verts_ooo[iju][1]+i) - k_gradient(verts_ooo[iju-1][0], verts_ooo[iju-1][1], verts_ooo[iju][0], verts_ooo[iju][1]))
                                    if rrr > math.pi:
                                        rrr = 2*math.pi - rrr
                                    for hi in range(len(min_k)):
                                        if min_k[hi] > rrr:
                                            re_pp_000.insert(hi,[verts_ooo[iju][0]+j, verts_ooo[iju][1]+i])
                                            min_k.insert(hi,rrr)
                                            min_k[-1] = min_k[0] + 0.018
                                            for i_jj in range(len(re_pp_000)):
                                                if min_k[-1] <= min_k[i_jj]:
                                                    re_pp_000 = re_pp_000[:i_jj]
                                                    min_k = min_k[:i_jj]
                                                    min_k.append(min_k[0] + 0.018)
                                                    break
                                            break
            if (len(re_pp_000) != (len(min_k)-1)):
                print("5")
                return [],[]
            max_lenl = -1
            re_pp_111= []
            # print("2.min_k: ",min_k[:-1])
            # print("re_pp_000: ", re_pp_000)
            for hii in re_pp_000:
                qq = distance_vv(hii[0],hii[1], re_pp[-1][0], re_pp[-1][1])
                if qq > max_lenl:
                    max_lenl = qq
                    re_pp_111 = hii
            if re_pp_111 != []:
                re_pp.append(re_pp_111)
        # print("len(verts_www),len(re_pp), len(verts_ooo), len(auau)",len(verts_www),len(re_pp), len(verts_ooo), auau)
        re_vv_transition = []
        if len(re_pp) < 3:
            return [], []
        
        # test__=True
        for ssr in range(len(re_pp)):
            if ssr == 0:
                ss_a = k_gradient(re_pp[-1][0],re_pp[-1][1],re_pp[0][0],re_pp[0][1])
                ss_b = k_gradient(re_pp[0][0],re_pp[0][1],re_pp[1][0],re_pp[1][1])
            elif ssr == (len(re_pp)-1):
                ss_a = k_gradient(re_pp[-2][0],re_pp[-2][1],re_pp[ssr][0],re_pp[ssr][1])
                ss_b = k_gradient(re_pp[ssr][0],re_pp[ssr][1],re_pp[0][0],re_pp[0][1])
            else:
                ss_a = k_gradient(re_pp[ssr-1][0],re_pp[ssr-1][1],re_pp[ssr][0],re_pp[ssr][1])
                ss_b = k_gradient(re_pp[ssr][0],re_pp[ssr][1],re_pp[ssr+1][0],re_pp[ssr+1][1])
            if (abs(ss_a - ss_b) < 2.9670597284 and abs(ss_a - ss_b) > 0.1745329252) or (abs(ss_a - ss_b) < 6.108652382 and abs(ss_a - ss_b) > 3.3161255788):
                re_vv_transition.append(re_pp[ssr])

            # test if is black_conotur_pixel
            
        #     if test__ == True:
        #         test_flag=True
        #         for i in self.this_black_contour:
        #             if i[0]==re_pp[ssr][0] and i[1]==re_pp[ssr][1]:
        #                 test_flag=False
        #                 break
        #         if test_flag==True:
        #             test__ = False
        # if test__ == False:
        #     print("$$$$$$$search_black_verts Error$$$$$$$$$$")
        # else:
        #     print("$Ok Ok Ok$")


        return re_pp, re_vv_transition
                        
                                



    def search_black_conotur_pixel(self, image):
        re = []
        for raw in range(len(image)):
            for i in range(len(image[raw])):
                if image[raw][i] == 0:
                    t = image[max(raw-1,0):min(raw+2, len(image)),max(i-1,0):min(i+2,len(image[raw]))]
                    if (t == 255).any():
                        re.append([int(i),int(raw)])
        return re


    # def search_near_white_pixel(self, verts)

    def select_door(self):
        def distance_door(x1,y1,x2,y2):
            return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
        # if self.this_flag_consistent_0 == False:
        #     tk.messagebox.showwarning(title='Error',message='Region not selected!!!')
        #     return
        # if self.this_flag_consistent_1 == True:
        #     tk.messagebox.showwarning(title='Error',message='Door has selected!!!')
        #     return
        print("\033[33m#Select Door *****\033[0m")
        print("Existing number of doors: ", len(self.this_json_data[self.this_map_id]['door']))
        plt.figure(4, dpi=160) 
        plt.imshow(self.this_image)
        try:
            pos=plt.ginput(0, False)#pos是像素坐标值(x,y)，不是矩阵索引，x是宽度
            print("Click point number: ", len(pos))
            gin_points = [[int(round(i[0])),int(round(i[1]))] for i in pos]
            se_points=[]
            for i_p in gin_points:
                if i_p not in se_points:
                    se_points.append(i_p)
            print("origianl pos (int):", se_points)
        except:
            tk.messagebox.showwarning(title='Error',message='Door failed:\n User exitd')
            plt.close(4)
            print("[] Door failed: User exitd")
            print("[] The number of doors: ", len(self.this_json_data[self.this_map_id]['door']))
            print("\033[33m ***** Door End\033[0m")
            return
        plt.close(4)
        if len(se_points) != 2:
            print("[] Door failed: Point len are erong")
            print("[] The number of doors: ", len(self.this_json_data[self.this_map_id]['door']))
            tk.messagebox.showwarning(title='Error',message='Door failed:\n Points are short')
        else:
            dis = [float('inf') for i in se_points]
            re = []
            bad_index = -1
            for index, point in enumerate(se_points):
                tmp_re = 0
                for x in self.this_black_contour:
                    if x not in re:
                        if dis[index] > distance_door(point[0],point[1],x[0], x[1]):
                            tmp_re = x
                            dis[index] = distance_door(point[0],point[1],x[0], x[1])
                re_x3i = []
                for x3i in self.this_black_verts:
                    if x3i not in re:
                        if distance_door(tmp_re[0],tmp_re[1],x3i[0], x3i[1]) < 4:
                            if len(re_x3i)==0:
                                re_x3i.append(x3i)
                            elif distance_door(tmp_re[0],tmp_re[1],x3i[0], x3i[1]) < distance_door(tmp_re[0],tmp_re[1],re_x3i[0][0], re_x3i[0][1]):
                                re_x3i.insert(0,x3i)
                            else:
                                for i in range(len(re_x3i)-1,-1,-1):
                                    if distance_door(tmp_re[0],tmp_re[1],x3i[0], x3i[1]) >= distance_door(tmp_re[0],tmp_re[1],re_x3i[i][0], re_x3i[i][1]):
                                        re_x3i.insert(i+1,x3i)
                                        break
                if re_x3i != []:
                    re.append(re_x3i[0])
                else:
                    bad_index = index
            if len(re) == 2: 
                re = self.tune_door(re)
                self.this_json_data[self.this_map_id]['door'].append(re)
                self.this_flag_consistent_1 = True
                self.this_flag_consistent_0 = False
                self.flag_area = True
                self.draw_target3(se_points)
                self.text.insert(tk.END,'\n door number --> {}'.format(len(self.this_json_data[self.this_map_id]['door'])))
                self.text.see(tk.END)
                print("[from] verts")
                print("New number of doors: ", len(self.this_json_data[self.this_map_id]['door']))
                
            else:
                pos = se_points
                dis_min_0 = float('inf')
                dis_min_1 = float('inf')
                vert0_s=[]
                vert1_s=[]
                for vert_test in self.this_json_data[self.this_map_id]['verts']:
                    if dis_min_0 > distance_door(vert_test[0], vert_test[1], pos[0][0], pos[0][1]):
                        dis_min_0 = distance_door(vert_test[0], vert_test[1], pos[0][0], pos[0][1])
                        vert0_s=vert_test
                    if dis_min_1 > distance_door(vert_test[0], vert_test[1], pos[1][0], pos[1][1]):
                        dis_min_1 = distance_door(vert_test[0], vert_test[1], pos[1][0], pos[1][1])
                        vert1_s=vert_test
                black_contour0_s_list=[(-1,-1),(-1,-1),(-1,-1)]
                black_contour1_s_list=[(-1,-1),(-1,-1),(-1,-1)]
                dis_min_0_list = [float('inf'), float('inf'), float('inf')]
                dis_min_1_list = [float('inf'), float('inf'), float('inf')]
                for vert_test in self.this_black_contour:
                    for i_0_ in range(3): 
                        if dis_min_0_list[i_0_] > distance_door(vert_test[0], vert_test[1], pos[0][0], pos[0][1]):
                            dis_min_0_list[i_0_]=distance_door(vert_test[0], vert_test[1], pos[0][0], pos[0][1])
                            black_contour0_s_list[i_0_]=vert_test
                        if dis_min_1_list[i_0_] > distance_door(vert_test[0], vert_test[1], pos[1][0], pos[1][1]):
                            dis_min_1_list[i_0_]=distance_door(vert_test[0], vert_test[1], pos[1][0], pos[1][1])
                            black_contour1_s_list[i_0_]=vert_test
                #这里有问题，上面的可能最后并不是三个距离最小的点，与点出现的顺序有关
                dis_min_0_l = float('inf')
                dis_min_1_l = float('inf')
                re_0=[]
                re_1=[]
                for i_0_ in range(3): 
                    if dis_min_0_l > distance_door(vert0_s[0],vert0_s[1],black_contour0_s_list[i_0_][0],black_contour0_s_list[i_0_][1]):
                        dis_min_0_l = distance_door(vert0_s[0],vert0_s[1],black_contour0_s_list[i_0_][0],black_contour0_s_list[i_0_][1])
                        re_0 = black_contour0_s_list[i_0_]
                    if dis_min_1_l > distance_door(vert1_s[0],vert1_s[1],black_contour1_s_list[i_0_][0],black_contour1_s_list[i_0_][1]):
                        dis_min_1_l = distance_door(vert1_s[0],vert1_s[1],black_contour1_s_list[i_0_][0],black_contour1_s_list[i_0_][1])
                        re_1 = black_contour1_s_list[i_0_]
                # re_0 = black_contour0_s_list[0]
                # re_1 = black_contour1_s_list[0]  #Take the nearest point directly
                if (re_0 != []) and (re_1 != []):
                    if len(re) == 1 and bad_index != -1:
                        if bad_index == 0 :
                            re_new_888 = self.tune_door([re_0,re[0]])
                            self.this_json_data[self.this_map_id]['door'].append(re_new_888)
                            print("[from] 1.contours 2.verts")
                        else:
                            re_new_888 = self.tune_door([re[0],re_1])
                            self.this_json_data[self.this_map_id]['door'].append(re_new_888)
                            print("[from] 1.verts 2.contours")
                        print("New number of doors: ", len(self.this_json_data[self.this_map_id]['door']))
                        self.text.insert(tk.END,'\n door number --> {}'.format(len(self.this_json_data[self.this_map_id]['door'])))
                        self.text.see(tk.END)
                        self.this_flag_consistent_1 = True
                        self.this_flag_consistent_0 = False
                        self.flag_area = True
                        self.draw_target3(se_points)
                    else:
                        print("[from] contours")
                        re_new_888 = self.tune_door([re_0,re_1])
                        self.this_json_data[self.this_map_id]['door'].append(re_new_888)
                        print("New number of doors: ", len(self.this_json_data[self.this_map_id]['door']))
                        self.this_flag_consistent_1 = True
                        self.this_flag_consistent_0 = False
                        self.flag_area = True
                        self.draw_target3(se_points)
                else:
                    print("Door points are wrong!!!")
                    print("[] The number of doors: ", len(self.this_json_data[self.this_map_id]['door']))
                    tk.messagebox.showwarning(title='Error',message='Door points are wrong!!!')
        
        print("\033[33m ***** Door End\033[0m")


            



    def select_area(self):
        print("\033[33m#Select Area *****\033[0m")
        print("Existing number of rooms: ", len(self.this_json_data[self.this_map_id]['areas']))
        plt.figure(2, dpi=160) 
        # plt.figure(2,figsize=(8,12),dpi=100) #figsize决定取点时显示图像的大小
        plt.imshow(self.this_image)
        try:
            pos=plt.ginput(0, False)#pos是像素坐标值(x,y)，不是矩阵索引，x是宽度
            print("Click point number: ", len(pos))
            gin_points = [[int(round(i[0])),int(round(i[1]))] for i in pos]
            se_points=[]
            for i_p in gin_points:
                if i_p not in se_points:
                    se_points.append(i_p)
            print("origianl pos (int):", se_points)
        except:
            tk.messagebox.showwarning(title='Error',message='Region failed:\n User exitd')
            plt.close(2)
            print("[] Region failed: User exitd")
            print("[] The number of rooms: ", len(self.this_json_data[self.this_map_id]['areas']))
            print("\033[33m ***** Area End\033[0m")
            return
        plt.close(2)
        if len(se_points) < 3:
            print("[] Region failed: Points are short")
            print("[] The number of rooms: ", len(self.this_json_data[self.this_map_id]['areas']))
            tk.messagebox.showwarning(title='Error',message='Region failed:\n Points are short')
        else:
            area = Polygon(se_points)
            polygon_pos = list(area.exterior.coords)
            polygon_pos = [[int(round(i[0])),int(round(i[1]))] for i in polygon_pos]
            # print("polyfon pos(int):", polygon_pos)
            se_points.append(se_points[0]) #se_points被加上了首尾重复点


            data_correctness = True


            # if len(polygon_pos)==len(se_points):
            #     for i in range(len(se_points)):
            #         if (polygon_pos[i][0] != se_points[i][0]) or (polygon_pos[i][1] != se_points[i][1]):
            #             data_correctness = False
            #             print("polygon_pos!=se_points")
            #             break
            # else:
            #     data_correctness = False
            #     print("*data Error*: len(polygon_pos)!=len(se_points)")


            area_is_valid = Polygon(se_points).is_valid
            if area_is_valid == False:
                data_correctness = False
                print("*data Error*: selected points polygon is not valid")
            else:
                test_max_index = -1
                for index_test, point_test in enumerate(se_points): 
                    test_max_index = index
                    if index_test == (len(se_points) - 1):
                        if self.poly_not_cross_open(se_points[:index_test], point_test, close_flag=1) == False:
                            data_correctness = False
                            print("*data Error*: Seleted polygon is crossed, close_flag=1!!!")
                            break
                    else:
                        if self.poly_not_cross_open(se_points[:index_test], point_test) == False:
                            data_correctness = False
                            print("*data Error*: Seleted polygon is crossed!!!")
                            break
                if test_max_index == -1:
                    data_correctness == False
                    print("*data Error*: se_point is None")

                area_this = area.area*1.0/self.meter2pixel/self.meter2pixel
                # print("area: ",area_this, "m2")
                if area_this<self.room_min or area_this > self.room_max:
                    data_correctness = False
                    print("The area of room: ", area_this)
                    print("*data Error*: selected points polygon area is not satisfied")

            if data_correctness == True:
                # cw_flag = Polygon(se_points).exterior.is_ccw
                # print('ccw:',cw_flag)
                # if cw_flag:
                #     se_points = se_points[::-1]
                # print("se_points: ", se_points)
                # auto_points = self.contour_adjustment(se_points) #auto_points有首尾重复点
                auto_points = self.contour_adjustment_verts_2(se_points) #auto_points有首尾重复点
                # print("auto_points: ", auto_points)
                if auto_points == []:
                    data_correctness = False
                    print("*data Error*: Automatic generation point set error")
                else:
                    cw_flag = Polygon(auto_points).exterior.is_ccw #进行顺逆时针统一处理
                    print('ccw:',cw_flag, "(if True, reverse)")
                    if cw_flag:
                        auto_points = auto_points[::-1]
                    area_room = Polygon(auto_points[:-1])
                    print("auto_points: ", auto_points)
                    area_this_room =  area_room.area*1.0/self.meter2pixel/self.meter2pixel
                    # print("area: ",area_this_room, "m2")
                    if area_this_room <self.room_min or area_this_room > self.room_max:
                        data_correctness = False
                        print("The area of room: ", area_this_room)
                        print("*data Error*: selected points polygon area is not satisfied")
                    area_room_is_valid = area_room.is_valid
                    if area_room_is_valid == False:
                        data_correctness = False
                        print("*data Error*: selected points polygon is valid")
                    if data_correctness == True:
                        for exist_area in self.this_json_data[self.this_map_id]['areas']: #检测当前标注区域与之前区域是否有大面积重复
                            area_other = Polygon(exist_area[:-1])
                            try:
                                intersection = area_room.intersection(area_other).area
                            except:
                                print("*data Error*: Intersection compute unsuccessfully")
                                data_correctness = False
                                break
                            print("intersection: ",intersection)
                            if intersection > min(area_other.area, area_room.area)*0.5:
                                print("*data Error*: It overlaps with the previous region")
                                data_correctness = False
                                break

            if data_correctness == True:
                # if self.this_flag_consistent_0 == True:
                #     tk.messagebox.showwarning(title='Error',message='Door not selected!!!')
                #     return
                # cw_flag = Polygon(se_points).exterior.is_ccw
                # print('ccw:',cw_flag)
                # if cw_flag:
                #     se_points = se_points[::-1]
                # auto_points = self.contour_adjustment(se_points)
                self.this_json_data[self.this_map_id]['areas'].append(auto_points) #auto_points作为最终结果，全是self.this_black_contour中的点（为像素为0的障碍物点）；其含义为像素坐标值[x,y]，x是宽度
                self.flag_area = True
                self.this_flag_consistent_0 = True
                self.this_flag_consistent_1 = False
                # self.draw_target()
                self.draw_target2(se_points)
                # print('[areas]:',self.this_json_data[self.this_map_id]['areas'])
                # print('result: ',self.this_json_data[self.this_map_id]['areas'])
                print("New number of rooms: ", len(self.this_json_data[self.this_map_id]['areas']))
                self.text.insert(tk.END,'\n room number --> {}'.format(len(self.this_json_data[self.this_map_id]['areas'])))
                self.text.see(tk.END)
            else:
                print("[] Polygon points are wrong!!!")
                tk.messagebox.showwarning(title='Error',message='Polygon points are wrong!!!')
                # print('len(polygon_pos,len(se_points)',len(polygon_pos),len(se_points))
                # print('self.this_json_data[self.this_map_id][areas]:',self.this_json_data[self.this_map_id]['areas'])
                print("[] The number of rooms: ", len(self.this_json_data[self.this_map_id]['areas']))
        print("\033[33m ***** Area End\033[0m")

    # def contour_adjustment(self, points):
    #     src = points[:(len(points)-1)]
    #     dis = [float('inf') for i in src]
    #     re = [0 for i in src]
    #     def distance(x1,y1,x2,y2):
    #         return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    #     for index, point in enumerate(src):
    #         for x in self.this_black_contour:
    #             if dis[index] > distance(point[0],point[1],x[0], x[1]):
    #                 re[index] = x
    #                 dis[index] = distance(point[0],point[1],x[0], x[1])
    #     re2=[]
    #     for i in re:
    #         if i not in re2:
    #             re2.append(i)

    #     re2.append(re2[0])
    #     return re2




    def contour_adjustment_verts_2(self, points): #第二个版本的contour_adjustment，带有verts
        neighbor_range = 2
        # 后期这个地方需要考虑是否还要从半径为3的圆内搜索，如果不要可以直接使用最近点，最近点不符合直接舍弃
        src = points[:(len(points)-1)]
        dis = [float('inf') for i in src]
        re = []
        def distance(x1,y1,x2,y2):
            return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
        def re_range_verts(target_l):
            new_transition_0 = []
            new_transition_no = []
            for i_src in target_l:
                for j_transition_0 in self.this_black_verts_re_vv_transition:
                    if i_src[0] == j_transition_0[0] and i_src[1] == j_transition_0[1]:
                        new_transition_0.append(i_src)
                    else:
                        new_transition_no.append(i_src)
            return new_transition_0 + new_transition_no

            
        for index, point in enumerate(src):
            tmp_re = 0
            for x in self.this_black_contour:
                if x not in re:
                    if dis[index] > distance(point[0],point[1],x[0], x[1]):
                        tmp_re = x
                        dis[index] = distance(point[0],point[1],x[0], x[1])
            # print("most close: ",tmp_re)
            re_x3i = []
            for x3i in self.this_black_verts:
                if x3i not in re:
                    if distance(tmp_re[0],tmp_re[1],x3i[0], x3i[1]) < 3*neighbor_range:
                            if len(re_x3i)==0:
                                re_x3i.append(x3i)
                            elif distance(tmp_re[0],tmp_re[1],x3i[0], x3i[1]) < distance(tmp_re[0],tmp_re[1],re_x3i[0][0], re_x3i[0][1]):
                                re_x3i.insert(0,x3i)
                            else:
                                for i in range(len(re_x3i)-1,-1,-1):
                                    if distance(tmp_re[0],tmp_re[1],x3i[0], x3i[1]) >= distance(tmp_re[0],tmp_re[1],re_x3i[i][0], re_x3i[i][1]):
                                        re_x3i.insert(i+1,x3i)
                                        break
            if re_x3i != []:
                fail_index = -1
                for index_tmp, i in enumerate(re_x3i):
                    if self.poly_not_cross_open(re, i) == True:
                        re.append(i)
                        break
                    else:
                        fail_index = index_tmp
                if fail_index == (len(re_x3i)-1):
                    print("all re_x3i are crossed")
                else:
                    continue


            # print("most close: ",tmp_re)
            if self.poly_not_cross_open(re, tmp_re) == True:
                re.append(tmp_re)
                # print("target search point: ", point)
                # print("most close: ",tmp_re)
                # print("re: ", re)
            else:
                print("search close points")
                close_points = []
                for x2 in self.this_black_contour:
                    # print("search close points")
                    if x2 not in re:
                        if 0 < distance(tmp_re[0],tmp_re[1],x2[0], x2[1]) < neighbor_range:
                            if len(close_points)==0:
                                close_points.append(x2)
                            elif distance(tmp_re[0],tmp_re[1],x2[0], x2[1]) < distance(tmp_re[0],tmp_re[1],close_points[0][0], close_points[0][1]):
                                close_points.insert(0,x2)
                            else:
                                for i in range(len(close_points)-1,-1,-1):
                                    if distance(tmp_re[0],tmp_re[1],x2[0], x2[1]) >= distance(tmp_re[0],tmp_re[1],close_points[i][0], close_points[i][1]):
                                        close_points.insert(i+1,x2)
                                        break
                if close_points != []:
                    print("target search point: ", point)
                    print("most close: ",tmp_re)
                    print("colse points: ", close_points)
                    print("re: ", re)
                    fail_index = -1
                    for index_tmp, x3 in enumerate(close_points):
                        if self.poly_not_cross_open(re, x3) == True:
                            re.append(x3)
                            break
                        else:
                            fail_index = index_tmp
                    if fail_index == (len(close_points)-1):
                        print("all close_points are crossed")
                        return []
                        
                else:
                    print("can not find close points")
                    return []
        if self.poly_not_cross_open(re, re[0], close_flag=1) == True:
            re.append(re[0])
            return re
        else:
            return []



    def contour_adjustment(self, points): #这是第一个版本的contour_adjustment，不带verts匹配的
        neighbor_range = 2
        # 后期这个地方需要考虑是否还要从半径为3的圆内搜索，如果不要可以直接使用最近点，最近点不符合直接舍弃
        src = points[:(len(points)-1)]
        dis = [float('inf') for i in src]
        re = []
        def distance(x1,y1,x2,y2):
            return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
        for index, point in enumerate(src):
            tmp_re = 0
            for x in self.this_black_contour:
                if x not in re:
                    if dis[index] > distance(point[0],point[1],x[0], x[1]):
                        tmp_re = x
                        dis[index] = distance(point[0],point[1],x[0], x[1])
            # print("most close: ",tmp_re)
            if self.poly_not_cross_open(re, tmp_re) == True:
                re.append(tmp_re)
                # print("target search point: ", point)
                # print("most close: ",tmp_re)
                # print("re: ", re)
            else:
                print("search close points")
                close_points = []
                for x2 in self.this_black_contour:
                    # print("search close points")
                    if x2 not in re:
                        if 0 < distance(tmp_re[0],tmp_re[1],x2[0], x2[1]) < neighbor_range:
                            if len(close_points)==0:
                                close_points.append(x2)
                            elif distance(tmp_re[0],tmp_re[1],x2[0], x2[1]) < distance(tmp_re[0],tmp_re[1],close_points[0][0], close_points[0][1]):
                                close_points.insert(0,x2)
                            else:
                                for i in range(len(close_points)-1,-1,-1):
                                    if distance(tmp_re[0],tmp_re[1],x2[0], x2[1]) >= distance(tmp_re[0],tmp_re[1],close_points[i][0], close_points[i][1]):
                                        close_points.insert(i+1,x2)
                                        break
                if close_points != []:
                    print("target search point: ", point)
                    print("most close: ",tmp_re)
                    print("colse points: ", close_points)
                    print("re: ", re)
                    fail_index = -1
                    for index_tmp, x3 in enumerate(close_points):
                        if self.poly_not_cross_open(re, x3) == True:
                            re.append(x3)
                            break
                        else:
                            fail_index = index_tmp
                    if fail_index == (len(close_points)-1):
                        print("all close_points are crossed")
                        return []
                        
                else:
                    print("can not find close points")
                    return []
        if self.poly_not_cross_open(re, re[0], close_flag=1) == True:
            re.append(re[0])
            return re
        else:
            return []


    # def contour_adjustment2(self, points):
    #     # 后期这个地方需要考虑是否还要从半径为3的圆内搜索，如果不要可以直接使用最近点，最近点不符合直接舍弃
    #     src = points[:(len(points)-1)]
    #     dis = [float('inf') for i in src]
    #     re = []
    #     def distance(x1,y1,x2,y2):
    #         return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    #     for index, point in enumerate(src):
    #         tmp_re = 0
    #         for x in self.this_black_contour:
    #             if x not in re:
    #                 if dis[index] > distance(point[0],point[1],x[0], x[1]):
    #                     tmp_re = x
    #                     dis[index] = distance(point[0],point[1],x[0], x[1])
    #         # print("target search point: ", point)
    #         # print("most close: ",tmp_re)
    #         # print("re: ", re)
    #         if self.poly_not_cross_open(re, tmp_re) == True:
    #             re.append(tmp_re)
    #         else:
    #             print("The nearest point is not suitable.")
    #             return []
    #     if self.poly_not_cross_open(re, re[0], close_flag=1) == True:
    #         re.append(re[0])
    #         return re
    #     else:
    #         return []


    def show_black_verts(self):
        plt.cla() 
        self.fig1.imshow(self.this_image)
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        data = self.this_black_verts
        x = [data[i][0] for i in range(len(data))]
        y = [data[i][1] for i in range(len(data))]
        x.append(x[0])
        y.append(y[0])
        self.fig1.scatter(x, y, color='b', linewidths = 8.0, marker="x")
        self.fig1.plot(x, y, color='lime', linewidth=1.0)
        data = self.this_black_verts_re_vv_transition
        x = [data[i][0] for i in range(len(data))]
        y = [data[i][1] for i in range(len(data))]
        self.fig1.scatter(x, y, color='red', linewidths = 8.0, marker="+")
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        self.canvs.draw()




    def draw_target(self, im = ''):
        if im == '':
            im = self.this_image
        plt.cla() 
        self.fig1.imshow(im)
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        data = self.this_json_data[self.this_map_id]['areas']
        for semantic_area in data:
            x = [semantic_area[i][0] for i in range(len(semantic_area))]
            y = [semantic_area[i][1] for i in range(len(semantic_area))]
            # x.append(x[0])
            # y.append(y[0])
            self.fig1.plot(x, y, color='lime', linewidth=3.0)
            self.fig1.scatter(x, y, color='b', linewidths = 8.0)
            polygon_for_c = Polygon(semantic_area)
            self.fig1.scatter(polygon_for_c.centroid.x, polygon_for_c.centroid.y, color='r', s=90, marker='+',linewidths = 8.0)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        self.canvs.draw()

    def draw_target2(self,  points2=[], im = ''):
        color_all = ['darksalmon', 'sienna', 'sandybrown', 'bisque', 'ran', 'gold', 'chartreuse', 'darkgreen', 'darkcyan', 'c',
                     'm', 'blue', 'g', 'deeppink', 'navy', 'fuchsia', 'lime', 'yellow', 'brown', 'peru']
        color_nn = 0
        if im == '':
            im = self.this_image
        plt.cla() 
        self.fig1.imshow(im)
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        data = self.this_json_data[self.this_map_id]['areas']
        tt_data = self.this_json_data[self.this_map_id]['door']
        num_xx = 0
        for semantic_area in data:
            x = [semantic_area[i][0] for i in range(len(semantic_area))]
            y = [semantic_area[i][1] for i in range(len(semantic_area))]
            # x.append(x[0])
            # y.append(y[0])
            # self.fig1.plot(x, y, color='lime', linewidth=3.0)
            self.fig1.plot(x, y, color='lime', linewidth=3.0)
            self.fig1.scatter(x, y, color='blue', linewidths = 8.0)
            polygon_for_c = Polygon(semantic_area)
            self.fig1.scatter(polygon_for_c.centroid.x, polygon_for_c.centroid.y, color='r', s=90, marker='+',linewidths = 8.0)
            # self.fig1.scatter(tt_data[0][0], tt_data[0][1], color=color_all[color_nn], linewidths = 8.0)
            # self.fig1.scatter(tt_data[1][0], tt_data[1][1], color=color_all[color_nn], linewidths = 8.0)
        
            # num_xx = num_xx + 1
            color_nn = color_nn + 1
        for door_pp in tt_data:
            self.fig1.scatter(door_pp[0][0], door_pp[0][1], color='red', linewidths = 10.0, marker='2')
            self.fig1.scatter(door_pp[1][0], door_pp[1][1], color='red', linewidths = 10.0, marker='2')
            self.fig1.plot([door_pp[0][0],door_pp[1][0]], [door_pp[0][1],door_pp[1][1]], color='red', linewidth=3.0)
            num_xx = num_xx + 1
        if points2 != []:
            x = [points2[i][0] for i in range(len(points2))]
            y = [points2[i][1] for i in range(len(points2))]
            x.append(x[0])
            y.append(y[0])
            self.fig1.plot(x, y, color='gold', linewidth=2.0)
            self.fig1.scatter(x, y, color='gold', linewidths = 4.0)     
            # if self.semantic_area[key][1] == 0:
            #     self.fig1.plot(x, y, color='r', linewidth=4.0)
            # if self.semantic_area[key][1] == 1:
            #     self.fig1.plot(x, y, color='blue', linewidth=4.0)
        # if self.points != [] and self.flag_area == True:
        #     x = [self.points[i][0] for i in range(len(self.points))]
        #     y = [self.points[i][1] for i in range(len(self.points))]
        #     x.append(x[0])
        #     y.append(y[0])
        #     self.fig1.plot(x, y, color='lime', linewidth=3.0)
        #     self.fig1.scatter(x, y, color='b', linewidths = 8.0)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        self.canvs.draw()

    def draw_target3(self,  points2=[], im = ''):
        color_all = ['darksalmon', 'sienna', 'sandybrown', 'bisque', 'ran', 'gold', 'chartreuse', 'darkgreen', 'darkcyan', 'c',
                     'm', 'blue', 'g', 'pink', 'deeppink', 'navy', 'fuchsia', 'lime', 'yellow', 'brown', 'peru']
        color_nn = 0
        if im == '':
            im = self.this_image
        plt.cla() 
        self.fig1.imshow(im)
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        data = self.this_json_data[self.this_map_id]['areas']
        tt_data = self.this_json_data[self.this_map_id]['door']
        num_xx = 0
        for semantic_area in data:
            x = [semantic_area[i][0] for i in range(len(semantic_area))]
            y = [semantic_area[i][1] for i in range(len(semantic_area))]
            # x.append(x[0])
            # y.append(y[0])
            # self.fig1.plot(x, y, color='lime', linewidth=3.0)
            self.fig1.plot(x, y, color='lime', linewidth=3.0)
            self.fig1.scatter(x, y, color='blue', linewidths = 8.0)
            
            # self.fig1.scatter(tt_data[0][0], tt_data[0][1], color=color_all[color_nn], linewidths = 8.0)
            # self.fig1.scatter(tt_data[1][0], tt_data[1][1], color=color_all[color_nn], linewidths = 8.0)
            polygon_for_c = Polygon(semantic_area)
            self.fig1.scatter(polygon_for_c.centroid.x, polygon_for_c.centroid.y, color='r', s=90, marker='+', linewidths = 8.0)
            # num_xx = num_xx + 1
            color_nn = color_nn + 1
        for door_pp in tt_data:
            self.fig1.scatter(door_pp[0][0], door_pp[0][1], color='red', linewidths = 10.0, marker='2')
            self.fig1.scatter(door_pp[1][0], door_pp[1][1], color='red', linewidths = 10.0, marker='2')
            self.fig1.plot([door_pp[0][0],door_pp[1][0]], [door_pp[0][1],door_pp[1][1]], color='red', linewidth=3.0)
            num_xx = num_xx + 1
        if points2 != []:
            self.fig1.scatter(points2[0][0], points2[0][1], color='peru', linewidths = 5.0)
            self.fig1.scatter(points2[1][0], points2[1][1], color='peru', linewidths = 5.0)  
            # if self.semantic_area[key][1] == 0:
            #     self.fig1.plot(x, y, color='r', linewidth=4.0)
            # if self.semantic_area[key][1] == 1:
            #     self.fig1.plot(x, y, color='blue', linewidth=4.0)
        # if self.points != [] and self.flag_area == True:
        #     x = [self.points[i][0] for i in range(len(self.points))]
        #     y = [self.points[i][1] for i in range(len(self.points))]
        #     x.append(x[0])
        #     y.append(y[0])
        #     self.fig1.plot(x, y, color='lime', linewidth=3.0)
        #     self.fig1.scatter(x, y, color='b', linewidths = 8.0)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        self.canvs.draw()




if __name__ == '__main__':
    semantic_annotation()
