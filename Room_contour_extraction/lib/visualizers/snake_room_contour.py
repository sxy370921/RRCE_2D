from lib.utils import img_utils, data_utils
from lib.utils.snake import snake_config
import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import cycle
import cv2
import math
from shapely.geometry import Polygon
from shapely.geometry import Point
import os
import json

mean = snake_config.mean
std = snake_config.std
num = 0

class Visualizer:
    def visualize_ex(self, output, batch):
        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
        ex = output['py']
        ex = ex[-1] if isinstance(ex, list) else ex
        ex = ex.detach().cpu().numpy() * snake_config.down_ratio

        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)

        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [254, 0, 0],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
        for i in range(len(ex)):
            color = next(colors).tolist()
            poly = ex[i]
            poly = np.append(poly, [poly[0]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], color=color)

        plt.show()


    def visualize_training_box(self, output, batch):
        global num
        # inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)) # with color augment (original version)
        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img_new_sxy(batch['inp'][0], mean, std).permute(1, 2, 0))   # without color augment (new version)
        box = output['detection'][:, :4].detach().cpu().numpy() * snake_config.down_ratio

        ex = output['py']
        ex = ex[-1] if isinstance(ex, list) else ex
        ex = ex.detach().cpu().numpy() * snake_config.down_ratio


        #Filtering and screening
        scores = output['detection'][:, 4].detach().cpu().numpy()
        valid_rooms = []
        checked_rooms = []
        for t in range(len(ex)):
            if t in checked_rooms:
                continue
            t_poly = ex[t]
            this_room = Polygon(t_poly).buffer(0.001)
            same_room_list=[[t,scores[t]]]
            for u in range(len(ex)):
                if u in checked_rooms:
                    continue
                if u != t:
                    u_poly = ex[u]
                    other_room = Polygon(u_poly).buffer(0.001)
                    intersection = this_room.intersection(other_room).area
                    if intersection > min(this_room.area, other_room.area)*0.5:
                        same_room_list.append([u,scores[u]])
            max_score = 0
            target_room = t
            for s in same_room_list:
                checked_rooms.append(s[0])
                if s[1]>max_score:
                    max_score = s[1]
                    target_room = s[0]
            # valid_rooms.append(target_room) # without area validation

            inp = np.array(inp)
            copy_inp = np.zeros(inp.shape[:2], np.uint8)
            copy_poly = []
            for i_p in ex[target_room]:
                copy_poly.append([int(i_p[0]), int(i_p[1])])
            copy_poly = np.array(copy_poly, dtype = np.int32)
            cv2.fillPoly(copy_inp, [copy_poly],255)
            # test_room = Polygon(ex[target_room])
            # max_h=math.ceil(np.max(ex[target_room][:,1]))
            # min_h=int(np.min(ex[target_room][:,1]))
            # max_w=math.ceil(np.max(ex[target_room][:,0]))
            # min_w=int(np.min(ex[target_room][:,0]))
            # if max_h > inp.shape[0]:
            #     max_h = inp.shape[0] - 1
            # if max_w > inp.shape[1]:
            #     max_w = inp.shape[1] - 1
            # if min_h < 0:
            #     min_h=0
            # if min_w < 0:
            #     min_w=0
            # room_area = 0
            # for i in range(min_h, max_h+1):
            #     for j in range(min_w, max_w+1):
            #         if Point(j,i).within(test_room) == True or Point(j,i).within(test_room.boundary) == True:
            #             if np.any(inp[i][j] > 252/255.0):
            #                 room_area = room_area + 1
            # if room_area * 0.05 * 0.05 > 1.5:
            #     valid_rooms.append(target_room)
                
            max_h=math.ceil(np.max(copy_poly[:,1]))
            min_h=int(np.min(copy_poly[:,1]))
            max_w=math.ceil(np.max(copy_poly[:,0]))
            min_w=int(np.min(copy_poly[:,0]))
            if max_h > inp.shape[0]:
                max_h = inp.shape[0] - 1
            if max_w > inp.shape[1]:
                max_w = inp.shape[1] - 1
            if min_h < 0:
                min_h=0
            if min_w < 0:
                min_w=0
            room_area = 0
            for i in range(min_h, max_h+1):
                for j in range(min_w, max_w+1):
                    if copy_inp[i][j]==255 and np.any(inp[i][j] > 252/255.0):
                            room_area = room_area + 1
            if room_area * 0.05 * 0.05 > 1.5:
                valid_rooms.append(target_room)

        # fig, ax = plt.subplots(1, figsize=(6, 6))
        # fig.tight_layout()
        # ax.axis('off')
        # ax.imshow(inp)

        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [0, 0, 255],
            [188, 189, 32],
            [26, 190, 207],
            [148,0,211],
            [0,191,255],
            [255,0,255],
            [255,20,147],
            [47,100,100],
            [209,146,117],
            [255,140,0],
            [85,107,47],
            [0,250,154],
            [25,25,112],
            [255,0,0],
            [244,264,96],
            [160,82,45]
        ]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
            
        inp = np.array(inp)
        new_display_output = np.zeros([inp.shape[0],inp.shape[1],3], np.uint8)
        new_display_output[:,:,:3]=inp*255
        for i in range(len(ex)):
            if i not in valid_rooms:
                continue
            
            color = next(colors).tolist()
            poly = ex[i]
            poly = np.append(poly, [poly[0]], axis=0)
            # ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=5)


            cv2.drawContours(new_display_output,[np.round(poly).astype(np.int32)],-1,[int(color[0]*255),int(color[1]*255),int(color[2]*255)], thickness=2)

            temp_img = np.zeros(new_display_output.shape[:2], np.uint8)
            cv2.fillPoly(temp_img, [poly.astype(np.int32)],255)
            max_h=math.ceil(np.max(poly[:,1]))
            min_h=int(np.min(poly[:,1]))
            max_w=math.ceil(np.max(poly[:,0]))
            min_w=int(np.min(poly[:,0]))
            if max_h > new_display_output.shape[0]:
                max_h = new_display_output.shape[0] - 1
            if max_w > new_display_output.shape[1]:
                max_w = new_display_output.shape[1] - 1
            if min_h < 0:
                min_h=0
            if min_w < 0:
                min_w=0
            color_alpha = (255/max(color)-1)*0.8+1
            for i in range(min_h, max_h+1):
                for j in range(min_w, max_w+1):
                    if temp_img[i][j] == 255 and np.any(new_display_output[i][j]>252):
                        new_display_output[i][j]=cv2.addWeighted(np.array([[new_display_output[i][j][0],new_display_output[i][j][1],new_display_output[i][j][2]]], np.uint8), 0.7 ,np.array([[int(color[0]*255),int(color[1]*255),int(color[2]*255)]], np.uint8), 0.3, 0)[0]
            
        cv2.imwrite('Room_contour_output/map{}_contours.png'.format(num),new_display_output)
        cv2.imshow('Out Image', new_display_output)
        cv2.waitKey(2000)
        # cv2.destroyAllWindows()
        # plt.savefig('Room_contour_output/{}.png'.format(num))
        # # plt.show()
        # plt.ion()
        # plt.pause(4)
        # plt.close()
        num = num + 1






    def load_groudtruth(self):
        out_json = os.path.join("/media/sxy/home_HDD/project/python/snake-master/data/sxy_room_segmentation/annotations", 'room_ctr_dataset_realmaps')
        out_json2 = os.path.join("/media/sxy/home_HDD/project/python/snake-master/data/sxy_room_segmentation/annotations", 'room_ctr_dataset_realmaps_correction')
        with open(os.path.join(out_json, "room_ctr_dataset_realmaps.json")) as json_file:
            main_annotions = json.load(json_file)
        with open(os.path.join(out_json2, "room_ctr_dataset_realmaps.json")) as json_file:
            addition_annotions = json.load(json_file)
        image_order = {}
        map_id_image = {}
        for i_ins in main_annotions['images']:
            image_order.update({i_ins['file_name']:[]})
            map_id_image.update({i_ins['id']:i_ins['file_name']})
        for i_con in main_annotions['annotations']:
            image_order[map_id_image[i_con['image_id']]].append([[i_con['segmentation'][0][i], i_con['segmentation'][0][i+1]] for i in range(0, len(i_con['segmentation'][0])-1,2)])
        
        
        image_order_addtion = {}
        map_id_image_addtion = {}
        for i_ins in addition_annotions['images']:
            image_order_addtion.update({i_ins['file_name']:[]})
            map_id_image_addtion.update({i_ins['id']:i_ins['file_name']})
        for i_con in addition_annotions['annotations']:
            image_order_addtion[map_id_image_addtion[i_con['image_id']]].append([[i_con['segmentation'][0][i], i_con['segmentation'][0][i+1]] for i in range(0, len(i_con['segmentation'][0])-1,2)])



        
        for new_rooms in image_order_addtion:
            if new_rooms in image_order:
                image_order[new_rooms] = image_order_addtion[new_rooms]
                print(new_rooms+" --- use data in correction")
            else:
                image_order.update({new_rooms:image_order_addtion[new_rooms]})
                print(new_rooms+" --- add new data")
        self.image_order=image_order
        self.all_prediction = 0
        self.all_rooms = 0
        self.all_free_tru = 0
        self.all_cont_tru = 0
        self.free_map_tru = 0
        self.cont_map_tru = 0


    def statistics(self, img_number):
        acc_free = 1.0*self.all_free_tru/self.all_prediction
        acc_cont = 1.0*self.all_cont_tru/self.all_prediction
        rec_free = 1.0*self.all_free_tru/self.all_rooms
        rec_cont = 1.0*self.all_cont_tru/self.all_rooms
        map_free_rec = 1.0*self.free_map_tru/img_number
        map_cont_rec = 1.0*self.cont_map_tru/img_number
        resulting_image = np.ones([90,2000,3], np.uint8)*220
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 0)
        thickness = 2
        data_pos_all_1 = (10, 40)
        data_pos_all_2 = (10, 80)
        text1 = "rooms: {}; predictions: {}; tfree: {}; tcont: {}; tm_free: {} tm_cont: {}".format(self.all_rooms, self.all_prediction, self.all_free_tru, self.all_cont_tru, self.free_map_tru, self.cont_map_tru)
        text2 = "f_acc: {:.3f}; c_acc: {:.3f}; f_rec: {:.3f} c_rec: {:.3f}; f_map: {:.3f}; c_map: {:.3f}".format(acc_free, acc_cont, rec_free, rec_cont, map_free_rec, map_cont_rec)
        cv2.putText(resulting_image, text1, data_pos_all_1, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(resulting_image, text2, data_pos_all_2, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.imwrite('Room_contour_evaluation/'+'Metric.png', resulting_image)


    def visualize(self, output, batch):
        self.visualize_training_box(output, batch)

