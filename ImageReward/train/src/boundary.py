
import os, json, random
from config.options import *
from config.utils import *
from config.learning_rates import get_learning_rate_scheduler
# os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
opts.BatchSize = opts.batch_size * opts.accumulation_steps * opts.gpu_num
torch.set_printoptions(threshold=10_000)

from InpaintRewardDataset import InpaintRewardDataset
from ImageReward import ImageReward
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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


def loss_func(reward, target):

    target = target.to(reward.device)
    
    squared_diffs = (reward - target) ** 2
    l2_loss = squared_diffs.mean() #+ (weight ** 2).sum() * 0.01


    return l2_loss, squared_diffs



def comput_V(fea):
    fea = torch.matmul(fea.t(), fea)
    I = torch.eye(fea.size(0)).to('cuda')
    result = fea + 0.01 * I

    return result



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

    test_dataset = InpaintRewardDataset("valid_all_nor", config['root_path'])
    test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, collate_fn=collate_fn if not opts.rank_pair else None)
    train_dataset = InpaintRewardDataset("train_all_nor", config['root_path'])
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn if not opts.rank_pair else None)
    model = ImageReward(config, device).to(device)
    print(len(train_loader), len(test_loader))
    

   
    if opts.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)

    mlp = 256
    fr = 0.7
    iters = 8
    if get_rank() == 0:
        
        model_path = '../Data/InpaintReward.pt' 
        state_dict = {k: v for k, v in torch.load(model_path).items()}
        new_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        model.load_state_dict(new_dict)
        model.eval()

        scores_by_img_id = {}
 
      
        last_feature_dim = model.mlp.last_layer_weight.shape[1]
        sum_features = torch.zeros((last_feature_dim, last_feature_dim)).to(device)

        with torch.no_grad():
            for step, batch_data_package in tqdm(enumerate(train_loader)):
          
                img_key = batch_data_package['img_id_v123']
                reward, feature = model(batch_data_package)
                sum_features += comput_V(feature[0])
        
        sum_features /= step
        V_inverse = torch.inverse(sum_features)
        os.makedirs("./tab_bound", exist_ok=True)
        torch.save(V_inverse, f'./tab_bound/V_inverse_{mlp}_{fr}.pt')
 
        V_values = {}
        V_inverse= torch.load(f'./tab_bound/V_inverse_{mlp}_{fr}.pt')

        with torch.no_grad():
            for step, batch_data_package in tqdm(enumerate(train_loader)):
               
                img_id = batch_data_package['img_id_v123'][0]
                rewards, last_feature = model(batch_data_package)
                
                value = torch.sum(torch.pow(torch.matmul(last_feature[0], V_inverse), 2))
                V_values[img_id] = value.item()
               
                scores = batch_data_package['score'] 
                # ranks = batch_data_package['rank'] 
                loss, _ = loss_func(rewards, scores)
                scores_by_img_id[img_id] = loss.item() 

            

                
        json_data = json.dumps(V_values, indent=4)
       
        with open(f'./tab_bound/V_values_{mlp}_{fr}.json', 'w') as file:
                file.write(json_data)

        json_data = json.dumps(scores_by_img_id, indent=4)
        with open(f'./tab_bound/scores_{mlp}_{fr}.json', 'w') as file:
                file.write(json_data)

        # test_loss = torch.cat(test_loss, 0)
        # print('Test Loss %6.5f | Acc %6.4f' % (torch.mean(test_loss), sum(acc_list) / len(acc_list)))

