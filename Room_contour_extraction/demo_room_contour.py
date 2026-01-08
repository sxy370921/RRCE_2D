import torch.utils.data as data
import glob
import os
import cv2
import numpy as np
from lib.utils.snake import snake_voc_utils_sxy, snake_config, visualize_utils
from lib.utils import data_utils
from lib.config import cfg
import tqdm
import torch
from lib.networks import make_network
from lib.utils.net_utils import load_network
from lib.visualizers import make_visualizer


class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

        if os.path.isdir(cfg.demo_path):
            self.imgs = glob.glob(os.path.join(cfg.demo_path, '*'))
            self.maps_number = len(self.imgs)
        elif os.path.exists(cfg.demo_path):
            self.imgs = [cfg.demo_path]
        else:
            raise Exception('NO SUCH FILE')

    def normalize_image(self, inp):
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - snake_config.mean) / snake_config.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def __getitem__(self, index):
        img = self.imgs[index]
        img = cv2.imread(img)

        # With data augmentation
        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_voc_utils_sxy.augment_sxy(
                img, 'test', snake_config.mean, snake_config.std
            )

        
        ret = {'inp': inp}
        meta = {'center': center, 'scale': scale, 'test': '', 'ann': '', 'name':self.imgs[index].split('/')[-1]}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.imgs)


def demo():
    print("param-s-1: ", snake_config.ct_score)
    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    dataset = Dataset()
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(dataset):
        batch['inp'] = torch.FloatTensor(batch['inp'])[None].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        visualizer.visualize(output, batch)
    print("param-s-2: ", snake_config.ct_score)


