import os, shutil
import torch.distributed as dist
from tensorboardX import SummaryWriter

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def make_path(config):
    return "update{}_l1_weight={}".format(config.rl_model_update, config.l1_weight)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, 0o777)

def visualizer(config):
    if get_rank() == 0:
        # save_path = make_path(config)
        filewriter_path = os.path.join(config.save_folder, config.exp_name)
        if os.path.exists(filewriter_path):   
            shutil.rmtree(filewriter_path)
        makedir(filewriter_path)
        writer = SummaryWriter(filewriter_path, comment='visualizer')
        return writer