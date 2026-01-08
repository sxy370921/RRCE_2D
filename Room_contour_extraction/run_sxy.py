from unittest.mock import mock_open
from lib.config import cfg, args
import numpy as np
import os

mo_path = {
'roomTrain':'data/model/room_contour',
'roomVal':'data/model/room_contour'
}
if cfg.train.dataset in mo_path:
    cfg.model_dir = mo_path[cfg.train.dataset]



def run_prediction():
    import demo_room_contour
    demo_room_contour.demo()






if __name__ == '__main__':
    globals()['run_'+args.type]()
