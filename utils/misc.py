import sys
import os
from pydoc import locate
import numpy as np
import torch as th
import logging
from datetime import datetime
import random


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def set_initial_seed(seed):
    if seed == -1:
        seed = random.randrange(2**32-1)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

    return seed


def string2class(string):
    c = locate(string)
    if c is None:
        raise ModuleNotFoundError('{} cannot be found!'.format(string))
    return c


def create_datatime_dir(base_dir):
    datetime_dir = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_dir = os.path.join(base_dir, datetime_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir


def get_logger(name, log_dir, file_name, write_on_console):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")

    if file_name is not None:
        # file logger
        fh = logging.FileHandler(os.path.join(log_dir, file_name), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if write_on_console:
        # console logger
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def prompt_before_overwrite(fpath):
    ans = True
    if os.path.exists(fpath):
        eprint('{} already exists! Overwrite? [y/N]'.format(fpath))
        ans = sys.stdin.readline().strip().lower()
        if ans == 'y' or ans == 'yes':
            ans = True
        elif ans == 'n' or ans == 'no':
            ans = False
        else:
            eprint('Answer not understood. The default is NO.')

    return ans


def path_exists_with_message(fpath):
    if os.path.exists(fpath):
        eprint('{} alredy exists'.format(fpath))
        return True
    else:
        return False
