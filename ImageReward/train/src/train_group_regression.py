
import os
from config.options import *
from config.utils import *
from config.learning_rates import get_learning_rate_scheduler
os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
opts.BatchSize = opts.batch_size * opts.accumulation_steps * opts.gpu_num


from InpaintRewardDataset import InpaintRewardDataset, InpaintRewardDatasetRegressionGroup
from ImageReward import ImageReward, ImageRewardRegressionGroup

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
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


def loss_func(reward, target, weight):
   
    # target = torch.zeros(reward.shape[0], dtype=torch.long).to(reward.device)
    target = target.to(reward.device)
    
    squared_diffs = (reward - target) ** 2
    l2_loss = squared_diffs.mean() + (weight ** 2).sum() * 0.01
    
    # reward_diff = reward[:, 0] - reward[:, 1]
    # acc = torch.mean((reward_diff > 0).clone().detach().float())
    accuracy = 0

    for i in range(len(reward)):    
        if reward[i][0] > reward[i][2]:
            accuracy += 1
        else:
            accuracy += 0
    accuracy /= len(reward) 
    
    return l2_loss, squared_diffs, accuracy


if __name__ == "__main__":
    
    if opts.std_log:
        std_log()
    
    if opts.distributed:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        init_seeds(opts.seed + local_rank)
        
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        init_seeds(opts.seed)
        
    
    writer = visualizer()

    train_dataset = InpaintRewardDatasetRegressionGroup("train_group_all_nor", config['root_path'])
    valid_dataset = InpaintRewardDatasetRegressionGroup("valid_group_all_nor", config['root_path'])
    test_dataset = InpaintRewardDatasetRegressionGroup("valid_group_all_nor", config['root_path'])
   
    
    if opts.distributed:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, sampler=train_sampler, drop_last=True, collate_fn=collate_fn if not opts.rank_pair else None)
    else:
        train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn if not opts.rank_pair else None)


    valid_loader = DataLoader(valid_dataset, batch_size=opts.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn if not opts.rank_pair else None)
        
    test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, drop_last=True, collate_fn=collate_fn if not opts.rank_pair else None)
   
    # Set the training iterations.
    opts.train_iters = opts.epochs * len(train_loader)
    steps_per_valid = len(train_loader) // opts.valid_per_epoch
    print("len(train_dataset) = ", len(train_dataset))
    print("train_dataset.iters_per_epoch = ", train_dataset.iters_per_epoch)
    print("len(train_loader) = ", len(train_loader))
    print("steps_per_valid = ", steps_per_valid)

    model = ImageRewardRegressionGroup(device).to(device)
    
    if opts.preload_path:
        print('model has been loaded!')
        model = preload_model(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.adam_beta1, opts.adam_beta2), eps=opts.adam_eps)
    scheduler = get_learning_rate_scheduler(optimizer, opts)
    if opts.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)



    best_loss = 1e9
    optimizer.zero_grad()
    
    losses = []
    acc_list = []
    stem = 0
    for epoch in range(opts.epochs):

        for step, batch_data_package in enumerate(train_loader):
            model.train()
            rewards, w = model(batch_data_package)
            if not opts.distributed:
                scores = torch.stack([batch_data_package[k]['score'] for k in range(len(batch_data_package))], dim=0)
                ranks = torch.stack([batch_data_package[k]['rank'] for k in range(len(batch_data_package))], dim=0)
            else:
                scores = batch_data_package['score']
               
            loss, loss_list, acc = loss_func(rewards, scores, w)
            # loss regularization
            loss = loss / opts.accumulation_steps
            # back propagation
            loss.backward()

            losses.append(loss_list)
            acc_list.append(acc)

            iterations = epoch * len(train_loader) + step + 1
            train_iteration = iterations / opts.accumulation_steps
            
            # update parameters of net
            if (iterations % opts.accumulation_steps) == 0:
                # optimizer the net
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                # train result print and log 
                if get_rank() == 0:
                    losses_log = torch.cat(losses, 0)
                    print('Iteration %d | Loss %6.5f | Acc %6.4f' % (train_iteration, torch.mean(losses_log), sum(acc_list) / len(acc_list)))
                    writer.add_scalar('Train-Loss', torch.mean(losses_log), global_step=train_iteration)
                    writer.add_scalar('Train-Acc', sum(acc_list) / len(acc_list), global_step=train_iteration)
                    
                losses.clear()
                acc_list.clear()
            
            # valid result print and log
            if (iterations % steps_per_valid) == 0:
                if get_rank() == 0:
                    model.eval()
                    valid_loss = []
                    valid_acc_list = []
                    with torch.no_grad():
                        for step, batch_data_package in enumerate(valid_loader):
                            rewards, w = model(batch_data_package)
                            if not opts.distributed:
                                scores = torch.stack([batch_data_package[k]['score'] for k in range(len(batch_data_package))], dim=0)
                                ranks = torch.stack([batch_data_package[k]['rank'] for k in range(len(batch_data_package))], dim=0)
                            else:
                                scores = batch_data_package['score']
                               
                            loss, loss_list, acc = loss_func(rewards, scores, w)
                            valid_loss.append(loss_list)
                            valid_acc_list.append(acc)
                
                    # record valid and save best model
                    valid_loss = torch.cat(valid_loss, 0)
                    print('Validation - Iteration %d | Loss %6.5f | Acc %6.4f' % (train_iteration, torch.mean(valid_loss), sum(valid_acc_list) / len(valid_acc_list)))
                    writer.add_scalar('Validation-Loss', torch.mean(valid_loss), global_step=train_iteration)
                    writer.add_scalar('Validation-Acc', sum(valid_acc_list) / len(valid_acc_list), global_step=train_iteration)
                        
                    # if torch.mean(valid_loss) < best_loss:
                    print("Best Val loss so far. Saving model")
                    best_loss = torch.mean(valid_loss)
                    print("best_loss = ", best_loss)
                    save_model(model, stem)
                    stem += 1

    # test model
    if get_rank() == 0:
        print("training done")
        print("test: ")
        model = load_model(model)
        model.eval()

        test_loss = []
        acc_list = []
        with torch.no_grad():
            for step, batch_data_package in enumerate(test_loader):
                rewards, w = model(batch_data_package)
                if not opts.distributed:
                    scores = torch.stack([batch_data_package[k]['score'] for k in range(len(batch_data_package))], dim=0)
                    ranks = torch.stack([batch_data_package[k]['rank'] for k in range(len(batch_data_package))], dim=0)
                else:
                    scores = batch_data_package['score']
                    
                loss, loss_list, acc = loss_func(rewards, scores,  w)
                test_loss.append(loss_list)
                acc_list.append(acc)

        test_loss = torch.cat(test_loss, 0)
        print('Test Loss %6.5f | Acc %6.4f' % (torch.mean(test_loss), sum(acc_list) / len(acc_list)))

