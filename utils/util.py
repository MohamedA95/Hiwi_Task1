import json
import torch
import pandas as pd
import torch.distributed as dist
import logging
import logging.config
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')

def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0

def get_logger(name=None,log_dir='saved/',test=False,verbosity=2):
    log_levels = {0: logging.WARNING,1: logging.INFO,2: logging.DEBUG}
    if is_master():
        log_config = 'logger/test_logger_config.json' if test else 'logger/train_logger_config.json'
        log_config = Path(log_config)
        if log_config.is_file():
            config = read_json(log_config)
            for _, handler in config['handlers'].items():
                if 'filename' in handler:
                    handler['filename'] = str(Path(log_dir) / handler['filename'])
            logging.config.dictConfig(config)
        else:
            print("Warning: logging configuration file is not found in {}.".format(log_config))
            logging.basicConfig(level=logging.INFO)
    logger=logging.getLogger(name)
    logger.setLevel(log_levels[verbosity])
    return logger

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, round(value,5))
        self._data.total[key] += round(value * n,5)
        self._data.counts[key] += n
        self._data.average[key] = round(self._data.total[key] / self._data.counts[key],5)

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
