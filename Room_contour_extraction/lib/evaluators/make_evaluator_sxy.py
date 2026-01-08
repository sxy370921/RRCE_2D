import imp
import os
from lib.datasets.dataset_catalog_sxy import DatasetCatalog_sxy


def _evaluator_factory(cfg):
    task = cfg.task
    data_source = DatasetCatalog_sxy.get(cfg.test.dataset)['id']
    module = '.'.join(['lib.evaluators', data_source, task])
    path = os.path.join('lib/evaluators', data_source, task+'_sxy.py')
    evaluator = imp.load_source(module, path).Evaluator(cfg.result_dir)
    return evaluator


def make_evaluator_sxy(cfg):
    if cfg.skip_eval:
        return None
    else:
        return _evaluator_factory(cfg)
