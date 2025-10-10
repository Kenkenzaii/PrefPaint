
import os
from tqdm import tqdm
from config.options import *
from config.utils import *
from config.learning_rates import get_learning_rate_scheduler
#os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
opts.BatchSize = opts.batch_size * opts.accumulation_steps * opts.gpu_num


from InpaintRewardDataset import InpaintRewardDatasetGroup, InpaintRewardDataset
from ImageReward import ImageRewardGroup, ImageReward

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.backends import cudnn

import sys

mse_loss = nn.MSELoss()


def std_log():
    if get_rank() == 0:
        save_path = make_path()
        makedir(config['log_base'])
        sys.stdout = open(os.path.join(config['log_base'], "{}.txt".format(save_path)), "w")


def init_seeds(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
       cudnn.deterministic = True
       cudnn.benchmark = False
    else:  # faster, less reproducible
       cudnn.deterministic = False
       cudnn.benchmark = True


def loss_func_my(reward, target, rank):
    
    # target = torch.zeros(reward.shape[0], dtype=torch.long).to(reward.device)
    target = target.to(reward.device)
    
    squared_diffs = (reward - target) ** 2
    l2_loss = squared_diffs.mean()
   
    if rank[0] == 1:
        if reward > 0:
            accuracy = 1
    elif rank[0] == 3:
        if reward < 0:
            accuracy = 1
    else:
        accuracy = 0.5
 
    
    return l2_loss, squared_diffs, accuracy


def loss_func(reward, weight):

    target = torch.zeros(reward.shape[0], dtype=torch.long).to(reward.device)
    loss_list = F.cross_entropy(reward, target, reduction='none')
    loss = torch.mean(loss_list) + (weight ** 2).sum() * 0.01
    
    # reward_diff = reward[:, 0] - reward[:, 1]
    # acc = torch.mean((reward_diff > 0).clone().detach().float())
    
    return loss, loss_list #, acc

# def setup(rank, world_size, master_port):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = master_port
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
def setup_dist(rank, local_rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)
    init_seeds(opts.seed + local_rank)

def main(local_rank, opts):
    if opts.std_log:
        std_log()
    if opts.distributed:
        rank = opts.node_rank * opts.gpu_num + local_rank
        world_size = opts.num_nodes * opts.gpu_num
        setup_dist(rank, local_rank, world_size, opts.master_addr, opts.master_port)
        device = torch.device("cuda", local_rank)
        init_seeds(opts.seed + local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        init_seeds(opts.seed)
    
    # writer = visualizer()

    valid_dataset = InpaintRewardDataset("valid_all_nor", config['root_path'])
    # valid_dataset = InpaintRewardDatasetGroup("valid_group_all_exp", config['root_path'])

    if opts.distributed:
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=opts.batch_size, sampler=valid_sampler, drop_last=True, collate_fn=collate_fn if not opts.rank_pair else None)
    else:
        valid_loader = DataLoader(valid_dataset, batch_size=opts.batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn if not opts.rank_pair else None)
    
    model = ImageReward(config, device).to(device)
    
    if opts.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    rate, iteration = 0.9, 2
    state_dict = {k: v for k, v in torch.load(f'/data1/kendong/ImageReward420l/train_my/checkpoint/Tab2_rank_256_{rate}_bs20_fix={rate}_lr=1e-05cosine/best_lr={iteration}.pt').items()}
    new_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    # state_dict = torch.load('/data1/kendong/ImageReward420l/train_my/checkpoint/A_128_001_bs4_ac1_gpus4_modifiedVit_v2_bs12_fix=0.7_lr=1e-05cosine/128_modified_vit.pt')
    model.load_state_dict(new_dict)


    reward_list = []
    test_loss = []
    acc_list = []
    print(len(valid_loader))
    
    if True: #get_rank() == 0:
        model.eval()
        with torch.no_grad():
            for step, batch_data_package in tqdm(enumerate(valid_loader)):
                # ranks = batch_data_package['rank'].to(device)
                rewards, w = model(batch_data_package)
                
                # res = torch.cat((rewards, ranks), dim=1)
                
                # rewards_nor = (rewards[0][0]-0.65823)/8.54
                # _, loss_list, acc = loss_func(rewards, w)
                # test_loss.append(loss_list)
                # acc_list.append(acc)
                reward_list.append(rewards)
                # if step % 500 == 0 and step != 0:
                #     print(torch.sum(torch.cat(reward_list, dim=0) > 0), torch.sum(torch.cat(reward_list, dim=0) < 0))
                    
        rewards = torch.cat(reward_list, dim=0)
        # print(rewards)
        # torch.save(rewards, 'rewards.pt')
    rewards = rewards[~torch.isnan(rewards)]
    print(torch.isnan(rewards).any()) 
    print(torch.isinf(rewards).any()) 
    print('Mean %6.5f | std %6.4f' % (torch.mean(rewards), torch.std(rewards).item()))
    # test_loss = torch.cat(test_loss, 0)
    # print('Test Loss %6.5f | Acc %6.4f' % (torch.mean(test_loss), sum(acc_list) / len(acc_list)))
    
   


if __name__ == "__main__":
    
    main(0, opts)
    # mp.spawn(main, args=(opts,), nprocs=opts.gpu_num, join=True)
    # if opts.std_log:
    #     std_log()
    
    # if opts.distributed:
    #     torch.distributed.init_process_group(backend="nccl")
    #     local_rank = torch.distributed.get_rank()
    #     torch.cuda.set_device(local_rank)
    #     device = torch.device("cuda", local_rank)
    #     init_seeds(opts.seed + local_rank)
    # else:
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     init_seeds(opts.seed)
        
    
    # writer = visualizer()


    # # valid_dataset = InpaintRewardDataset("all_nor", config['root_path'])
    # valid_dataset = InpaintRewardDatasetGroup("valid_group_all_exp", config['root_path'])
    # if opts.distributed:
    #     valid_sampler = DistributedSampler(valid_dataset)
    #     valid_loader = DataLoader(valid_dataset, batch_size=opts.batch_size, sampler=valid_sampler, drop_last=True, collate_fn=collate_fn if not opts.rank_pair else None)
    # else:
    #     valid_loader = DataLoader(valid_dataset, batch_size=opts.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn if not opts.rank_pair else None)

    # model = ImageRewardGroup(device).to(device)
    
    # if opts.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    # state_dict = {k: v for k, v in torch.load('/data1/kendong/ImageReward420l/train_my/checkpoint/A_128_bs4_ac1_gpus5_epoch100_bs20_fix=0.7_lr=1e-05cosine/best_lr=11.pt').items()}
    # new_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    # model.load_state_dict(new_dict)

    # # state_dict = model.load_state_dict(torch.load('/data1/kendong/ImageReward420l/train_my/checkpoint/A_128_bs4_ac1_gpus5_epoch100_bs20_fix=0.7_lr=1e-05cosine/best_lr=11.pt'), strict=False)
    # # new_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    # # model.load_state_dict(new_dict)

    # # if opts.preload_path:
    # #     model = preload_model(model)
    


    # reward_list = []
    # test_loss = []
    # acc_list = []
    # print(len(valid_loader))
    # # valid result print and log
    # if True: #get_rank() == 0:
    #     model.eval()
    #     valid_loss = []
    #     valid_acc_list = []
    #     with torch.no_grad():
    #         for step, batch_data_package in tqdm(enumerate(valid_loader)):
    #             rewards, w = model(batch_data_package)
    #             print(rewards)
    #             # rewards_nor = (rewards[0][0]-0.65823)/8.54
    #             loss, loss_list, acc = loss_func(rewards, w)
    #             test_loss.append(loss_list)
    #             acc_list.append(acc)
    #             reward_list.append(rewards)
              
              

    #     rewards = torch.cat(reward_list, dim=0)
    #     # tensor_str = rewards.numpy().tolist()
    #     # with open('tensor_data.txt', 'w') as file:
    #     #     for i in range(0, len(reward_list), 10):
    #     #         line_data = reward_list[i:i+10]
    #     #         file.write(' '.join(map(str, line_data)) + '\n')
   
    
    # rewards = rewards[~torch.isnan(rewards)]
    # print(torch.isnan(rewards).any()) 
    # print(torch.isinf(rewards).any()) 
  
    # # rewards = torch.cat(rewards, 0)
    # print('Mean %6.5f | std %6.4f' % (torch.mean(rewards), torch.std(rewards).item()))
    # test_loss = torch.cat(test_loss, 0)
    # print('Test Loss %6.5f | Acc %6.4f' % (torch.mean(test_loss), sum(acc_list) / len(acc_list)))
    
   