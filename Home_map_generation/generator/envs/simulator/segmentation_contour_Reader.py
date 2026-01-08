import json, cv2, os
import numpy as np




class contour_Reader():
    def __init__(self, json_path, meter2pixel):
        self.json_prefix = json_path      # json data dir
        self.meter2pixel = meter2pixel      # meter to pixel. E.g. meter2pixel=100 represents 1 meter
                                            # equals 100 pixels in the map
        self.border_pad = 8                 # border padding for each side

        self.json_data = {}                 # house info read from json file
        self.cnt_map = []                   # contour map, 0-obstacle, 255-free space

        self.segmentation_data = []

        self.map_boundary_points = []



    def search_black_conotur_pixel(self, image):
        re = []
        for raw in range(len(image)):
            for i in range(len(image[raw])):
                if image[raw][i] == 0:
                    t = image[max(raw-1,0):min(raw+2, len(image)),max(i-1,0):min(i+2,len(image[raw]))]
                    if (t == 255).any():
                        re.append([int(i),int(raw)])
        return re


    def check_contour_points(self, map,contour):
        self.map_boundary_points = self.search_black_conotur_pixel(map)
        nn = 0
        for area_points in contour:
            for point in area_points:
                nn = nn + 1
                if map[point[1],point[0]] == 0:
                    if point not in self.map_boundary_points:
                        return False
                else:
                    return False
        print("The number of checked points: ", nn)
        return True



    def read_json(self, file_name):
        """
        Read json file, generate and return contour map (cnt_map) and room type map (tp_map)
        :param file_name: name of a json file, e.g. 12345678.json
        :return: cnt_map, tp_map
        """
        # print("Processing ", file_name)

        with open(self.json_prefix + '/'+file_name.split('.')[0] + '.json') as json_file:
            self.json_data = json.load(json_file)

        # Draw the contour
        verts = (np.array(self.json_data['verts']) * self.meter2pixel).astype(np.int)
        x_max, x_min, y_max, y_min = np.max(verts[:,0]), np.min(verts[:,0]), np.max(verts[:, 1]), np.min(verts[:,1])
        self.cnt_map = np.zeros((y_max - y_min + self.border_pad * 2+1,
                        x_max - x_min + self.border_pad * 2+1))

        verts[:, 0] = verts[:, 0] - x_min + self.border_pad
        verts[:, 1] = verts[:, 1] - y_min + self.border_pad
        cv2.drawContours(self.cnt_map, [verts], 0, 255, -1)

        self.segmentation_data = self.point_transformer_json2image(self.json_data['areas'], x_min=x_min, y_min=y_min,border_pad= self.border_pad)

        self.door_data = self.point_transformer_json2image(self.json_data['door'], x_min=x_min, y_min=y_min,border_pad= self.border_pad)
        if self.check_contour_points(self.cnt_map, self.segmentation_data):
            return self.cnt_map.copy(), self.segmentation_data.copy(), self.map_boundary_points.copy(), self.door_data.copy(), True
        else:
            return self.cnt_map.copy(), self.segmentation_data.copy(), self.map_boundary_points.copy(), self.door_data.copy(), False
        # self.display(self.cnt_map)


        

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


if __name__ == '__main__':
    json_prefix = './resource/json/'

    reader = contour_Reader(json_prefix)
    files = os.listdir(json_prefix)
    for file in files:
        reader.read_json(file)

