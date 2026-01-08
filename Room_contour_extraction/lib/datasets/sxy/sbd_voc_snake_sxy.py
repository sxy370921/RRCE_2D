import os
from lib.utils.snake import snake_voc_utils_sxy, snake_config, visualize_utils
import cv2
import numpy as np
import math
from lib.utils import data_utils
import torch.utils.data as data
from pycocotools.coco import COCO
from lib.config import cfg


class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split):
        print('\033[1;31;47m sbd_voc_snake_sxy V2\033[0m')
        super(Dataset, self).__init__()
        self.logs_num = 0
        # self.logs_dir = "/media/sxy/home_HDD/project/python/snake-master/data/sxy_room_segmentation/interactiva_semantic_map/pic/deepsnake_dataset_augment"

        print('\033[1;31;47m dataset - img_path: {}\033[0m'.format(data_root))
        print('\033[1;31;47m dataset - ann_path: {}\033[0m'.format(ann_file))
        self.data_root = data_root
        self.split = split

        self.coco = COCO(ann_file)
        print('\033[1;31;47m dataset -img_lenth: {}\033[0m'.format(len(self.coco.getImgIds())))
        self.anns = np.array(sorted(self.coco.getImgIds()))
        self.anns = self.anns[:500] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}
        # print("Dataset init: ",'getImgIds(), ',self.coco.getImgIds(),'self.coco.getCatIds(), ',self.coco.getCatIds()) #test_sxy

    def process_info(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)
        path = os.path.join(self.data_root, self.coco.loadImgs(int(img_id))[0]['file_name'])
        return anno, path, img_id

    def read_original_data(self, anno, path):
        img = cv2.imread(path)
        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno]
        cls_ids = [self.json_category_id_to_contiguous_id[obj['category_id']] for obj in anno]
        return img, instance_polys, cls_ids

    def transform_original_data(self, instance_polys, flipped, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = [poly.reshape(-1, 2) for poly in instance]

            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = snake_voc_utils_sxy.transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_
    
    def transform_original_data_sxy(self, instance_polys, trans_output, inp_out_hw, img_origin):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = [poly.reshape(-1, 2) for poly in instance]
            polys = snake_voc_utils_sxy.transform_polys_sxy(polys, trans_output, output_h, output_w, img_origin)
            instance_polys_.append(polys)
        return instance_polys_



    def get_valid_polys(self, instance_polys, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            polys = snake_voc_utils_sxy.filter_tiny_polys(instance)
            polys = snake_voc_utils_sxy.get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            instance_polys_.append(polys)
        return instance_polys_

    def get_extreme_points(self, instance_polys):
        extreme_points = []
        for instance in instance_polys:
            points = [snake_voc_utils_sxy.get_extreme_points(poly) for poly in instance]
            extreme_points.append(points)
        return extreme_points

    def prepare_detection(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        ct = np.round(ct).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

        x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]

        return decode_box

    def prepare_detection_(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        box_ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)

        x_min_int, y_min_int = int(x_min), int(y_min)
        h_int, w_int = math.ceil(y_max - y_min_int) + 1, math.ceil(x_max - x_min_int) + 1
        max_h, max_w = ct_hm.shape[0], ct_hm.shape[1]
        h_int, w_int = min(y_min_int + h_int, max_h) - y_min_int, min(x_min_int + w_int, max_w) - x_min_int

        mask_poly = poly - np.array([x_min_int, y_min_int])
        mask_ct = box_ct - np.array([x_min_int, y_min_int])
        ct, off, xy = snake_voc_utils_sxy.prepare_ct_off_mask(mask_poly, mask_ct, h_int, w_int)

        xy += np.array([x_min_int, y_min_int])
        ct += np.array([x_min_int, y_min_int])

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

    def prepare_init(self, box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, h, w):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        img_init_poly = snake_voc_utils_sxy.get_init(box)
        img_init_poly = snake_voc_utils_sxy.uniformsample(img_init_poly, snake_config.init_poly_num)
        can_init_poly = snake_voc_utils_sxy.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)
        img_gt_poly = extreme_point
        can_gt_poly = snake_voc_utils_sxy.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        i_it_4pys.append(img_init_poly)
        c_it_4pys.append(can_init_poly)
        i_gt_4pys.append(img_gt_poly)
        c_gt_4pys.append(can_gt_poly)

    def prepare_evolution(self, poly, extreme_point, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        octagon = snake_voc_utils_sxy.get_octagon(extreme_point)
        img_init_poly = snake_voc_utils_sxy.uniformsample(octagon, snake_config.poly_num)
        can_init_poly = snake_voc_utils_sxy.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)

        img_gt_poly = snake_voc_utils_sxy.uniformsample(poly, len(poly) * snake_config.gt_poly_num)
        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
        can_gt_poly = snake_voc_utils_sxy.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)

    def prepare_merge(self, is_id, cls_id, cp_id, cp_cls):
        cp_id.append(is_id)
        cp_cls.append(cls_id)

    def __getitem__(self, index):
        # print("***__getitem__ of Dataset:{}***".format(index)) #test_sxy
        ann = self.anns[index]

        # print("__getitem__", index, ann)

        anno, path, img_id = self.process_info(ann)
        # print("anno: ",anno) #test_sxy
        img, instance_polys, cls_ids = self.read_original_data(anno, path) #img是图像，instance_polys是边界点，cls_ids是语义类别
        # print("instance_polys #1: ",instance_polys,'\n','cls_ids: ',cls_ids) #test_sxy

        height, width = img.shape[0], img.shape[1]

        # #No data augmentation but color augment (original version)
        # orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
        #     snake_voc_utils_sxy.augment(
        #         img, 'test',
        #         snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
        #         snake_config.mean, snake_config.std, instance_polys
        #     )


        # # No data augmentation
        # orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
        #     snake_voc_utils_sxy.augment(
        #         img, 'test',
        #         snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
        #         snake_config.mean, snake_config.std, instance_polys
        #     )
        # instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw)
        

        # With data augmentation
        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_voc_utils_sxy.augment_sxy(
                img, self.split, snake_config.mean, snake_config.std
            )
        instance_polys = self.transform_original_data_sxy(instance_polys, trans_output, inp_out_hw, img)
        
        # # logs
        # test_unit = np.array([[1, 0, 0], [0, 1, 0]],dtype=np.float64)
        # if np.any(trans_input != test_unit):
        #     if self.logs_num < 100:
        #         show_inp = np.zeros([512,512,3])
        #         for i_s in range(512):
        #             for j_s in range(512):
        #                 show_inp[i_s][j_s] = [inp[0][i_s][j_s]*255, inp[1][i_s][j_s]*255, inp[2][i_s][j_s]*255]
        #         self.logs_num = self.logs_num + 1
        #         # print("inp.shape", inp.shape)
        #         cv2.imwrite(self.logs_dir+"/inp{}.png".format(self.logs_num),show_inp)
        #         outp = cv2.warpAffine(img,trans_output,(512//4,512//4),flags=cv2.INTER_NEAREST,borderValue=[220, 220, 220])
        #         cv2.imwrite(self.logs_dir+"/out{}.png".format(self.logs_num),outp)
                
        #         # print(instance_polys)
        #         # print(np.array(instance_polys))
        #         instance_list = []
        #         for i_p in instance_polys:
        #             instance_list.append(i_p[0])
        #         # print(np.array(instance_list,dtype=object))

        #         cv2.drawContours(outp,instance_list,-1,[255,0,0])
        #         cv2.imwrite(self.logs_dir+"/out_c_{}.png".format(self.logs_num),outp)




        # print("instance_polys #2: ",instance_polys,'\n','cls_ids: ',cls_ids) #test_sxy
        instance_polys = self.get_valid_polys(instance_polys, inp_out_hw)
        # print("instance_polys #3: ",instance_polys,'\n','cls_ids: ',cls_ids) #test_sxy
        extreme_points = self.get_extreme_points(instance_polys)
        # print("extreme_points: ",extreme_points,'\n','cls_ids: ',cls_ids) #test_sxy

        # print('trans_input: ', trans_input) #test_sxy
        # print('trans_output: ', trans_output) #test_sxy

        # print('inp_out_hw, inp_out_hw[2:]:',inp_out_hw, inp_out_hw[2:]) #test_sxy
        # detection
        output_h, output_w = inp_out_hw[2:]
        ct_hm = np.zeros([cfg.heads.ct_hm, output_h, output_w], dtype=np.float32)
        wh = []
        ct_cls = []
        ct_ind = []

        # init
        i_it_4pys = []
        c_it_4pys = []
        i_gt_4pys = []
        c_gt_4pys = []

        # evolution
        i_it_pys = []
        c_it_pys = []
        i_gt_pys = []
        c_gt_pys = []

        # print("\033[31m 1. ct_hm shape: {}\033[0m".format(ct_hm.shape))#test_sxy
        for i in range(len(anno)):
            cls_id = cls_ids[i]
            instance_poly = instance_polys[i]
            instance_points = extreme_points[i]

            # print('len(instance_poly): ',len(instance_poly)) #test_sxy
            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                extreme_point = instance_points[j]

                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue

                self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
                self.prepare_init(bbox, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h, output_w)
                self.prepare_evolution(poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys)

        # print("\033[31m 2. ct_hm shape: {}\033[0m".format(ct_hm.shape))#test_sxy
        # print('ct_hm shape: ',ct_hm.shape) #test_sxy
        # print("\033[31m inp shape: {}\033[0m".format(inp.shape))#test_sxy
        # print("\033[31m wh: {}, ct_cls: {}, ct_ind: {}\033[0m".format(wh,ct_cls,ct_ind))#test_sxy
        ret = {'inp': inp}
        detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
        evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
        ret.update(detection)
        ret.update(init)
        ret.update(evolution)

        # print("\033[31m i_it_4pys: {}, i_gt_4pys: {}, i_it_pys: {}, i_gt_pys: {}\033[0m".format(i_it_4pys[0].shape, i_gt_4pys[0].shape, i_it_pys[0].shape, i_gt_pys[0].shape))#test_sxy
        # print("\033[31m #########\033[0m")#test_sxy
        # visualize_utils.visualize_snake_detection(orig_img, ret)
        # visualize_utils.visualize_snake_evolution(orig_img, ret)

        # print('ct_hm: ',ct_hm) #test_sxy
        # print('ret: ', ret) #test_sxy
        # print('i_gt_4pys: ',i_gt_4pys) #test_sxy
        # print('c_gt_4pys: ',c_gt_4pys) #test_sxy
        # print('i_gt_pys: ',i_gt_pys) #test_sxy
        # print('c_gt_pys: ',c_gt_pys) #test_sxy
        # print('detection,ct_hm.shape: ',detection, ct_hm.shape) #test_sxy
        # print('len(i_it_4pys), len(c_it_4pys), len(i_gt_4pys), len(c_gt_4pys),len(i_it_pys), len(c_it_pys), len(i_gt_pys),len(c_gt_pys): ', np.array(i_it_4pys).shape, np.array(c_it_4pys).shape, np.array(i_gt_4pys).shape, np.array(c_gt_4pys).shape,np.array(i_it_pys).shape, np.array(c_it_pys).shape, np.array(i_gt_pys).shape,np.array(i_gt_pys).shape) #test_sxy

        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'ct_num': ct_num}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.anns)

