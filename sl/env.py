import os
import shutil

import base.base_utils as the_utils

def setup(opt):
    cur_time = the_utils.cur_time_str()
    checkpoint_dir = os.path.join(opt.log_dir, cur_time, 'checkpoints')
    record_dir = os.path.join(opt.log_dir, cur_time, 'record')

    the_utils.mkdir(opt.log_dir)
    remove_useless_logs(opt.log_dir)
    the_utils.mkdir(checkpoint_dir)
    the_utils.mkdir(record_dir)
    return checkpoint_dir, record_dir


def remove_useless_logs(log_dir):
    dirnames = os.listdir(log_dir)
    dir_paths = [os.path.join(log_dir, dirname) for dirname in dirnames]

    for dir_path in dir_paths:
        checkpoint_dir = os.path.join(dir_path, 'checkpoints')
        if len(os.listdir(checkpoint_dir)) == 0:
            shutil.rmtree(dir_path)
            print(f'Remove useless log {dir_path}')


def remove_useless_cluster_logs(cluster_output_dir):
    dirnames = os.listdir(cluster_output_dir)
    full_dirs = [os.path.join(cluster_output_dir, dirname) for dirname in dirnames]
    for full_dir in full_dirs:
        items = os.listdir(full_dir)
        if len(items) == 0:
            shutil.rmtree(full_dir)
            print(f'Remove useless cluster output directories {full_dir}')


def remove_useless_records(record_dir):
    record_names = os.listdir(record_dir)
    record_paths = [os.path.join(record_dir, record_name) for record_name in record_names]

    for path in record_paths:
        count = -1
        with open(path, 'r') as f:
            count = len(f.readlines())

        if count == 0 or count == 1:
            os.remove(path)
            print(f'Remove useless record {path}')
