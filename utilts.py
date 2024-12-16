import os, glob
import torch.distributed as dist
from tensorboardX import SummaryWriter
from collections import deque


class MovingAverageStack:
    def __init__(self, size):
        self.stack = deque(maxlen=size)
        self.sum = 0.0

    def push(self, value):
        if len(self.stack) == self.stack.maxlen:
            self.sum -= self.stack.popleft()
        self.stack.append(value)
        self.sum += value
        return self.get_average()

    def get_average(self):
        if not self.stack:
            return 0  
        return self.sum / len(self.stack)



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
        makedir(filewriter_path)
        # event_files = glob.glob(os.path.join(filewriter_path, 'event*'))
        # for file_path in event_files:
        #     try:
        #         os.remove(file_path)
        #         print(f"Deleted file: {file_path}")
        #     except OSError as e:
        #         print(f"Error: {file_path} : {e.strerror}")
        
        writer = SummaryWriter(filewriter_path, comment='visualizer')
        return writer

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