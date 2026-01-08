import os
import imp


def make_visualizer(cfg):
    module = '.'.join(['lib.visualizers', cfg.task+'_room_contour'])
    path = os.path.join('lib/visualizers', cfg.task+'_room_contour'+'.py')
    visualizer = imp.load_source(module, path).Visualizer()
    return visualizer
