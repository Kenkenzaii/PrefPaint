

import os
import json
import math
import torch
from torch.utils.data import Dataset
from config.utils import *
from config.options import *
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from transformers import BertTokenizer
import clip
import numpy as np
from torchvision import transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer


class InpaintAlignmentDataset(Dataset):
    def __init__(self, dataset, root_path):

        self.inpaint_imgs = None
        self.mask_imgs = None
        self.root_path = root_path # '/data/kendong/Diffusions/joint-rl-diffusion/data/ade20k/merge'
        
        self.clip, self.preprocess = clip.load("ViT-B/32", device="cuda")  #clip.load(config['clip_model'], device="cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_path = os.path.join(config['data_base'], f"{dataset}.json")
        self.transform = transforms.ToTensor()
        self.dataset_path = os.path.join(config["data_base"], f"{dataset}.txt")
   
        
        with open(self.dataset_path, 'r') as txt_file:
            self.data_txt = [line.strip() for line in txt_file.readlines()]

        self.data = [
            {
                "img_id": key, 
            }
            for key  in self.data_txt
        ]
        # with open(self.dataset_path, "r") as f:
        #         self.data_json = json.load(f)
        # self.data = [{'img_id': key.rsplit('_', 1)[0], 'img_id_v123': key, 'inpaint_img': key, 'score': value[0], 'rank': value[1]} for key, value in self.data_json.items()]
        
        self.get_fileinfo() 

        self.iters_per_epoch = int(math.ceil(len(self.data)*1.0/opts.batch_size))

    def get_fileinfo(self):
        inpaint_images_list = []
        img_ids_list = []
    
        for item in self.data:
            inpaint_images_list.append(item['inpaint_img'])
            img_ids_list.append(item['img_id'])
   
        self.inpaint_imgs = [os.path.join(self.root_path, 'inpaint', name) for name in inpaint_images_list]
        self.img_ids = img_ids_list 
        self.mask_imgs = [os.path.join(self.root_path, 'masks_color', name.rsplit('_', 1)[0]+'_mask_color') for name in inpaint_images_list]
        
    def get_file(self, index):
        inpaint = Image.open(self.inpaint_imgs[index] + '.jpg')
        mask_rgb = Image.open(self.mask_imgs[index] + '.jpg')
        inpaint = self.preprocess(inpaint).unsqueeze(0)
        mask_rgb = self.preprocess(mask_rgb).unsqueeze(0)
        img_id = self.img_ids[index]

        data = {
            'img_id': img_id,
            "inpaint": inpaint,
            'mask_rgb': mask_rgb,
    
        }
        return data
        

    def __getitem__(self, index):
        return self.get_file(index)

    def __len__(self):
        return len(self.data)
    

class InpaintRewardDataset(Dataset):
    def __init__(self, dataset, root_path):

        self.inpaint_imgs = None
        self.mask_imgs = None
        self.scores = None
        self.ranks = None
        self.root_path = root_path 
        
        self.clip, self.preprocess = clip.load("ViT-B/32", device="cuda")  #clip.load(config['clip_model'], device="cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_path = os.path.join(config['data_base'], f"{dataset}.json")
        with open(self.dataset_path, "r") as f:
                self.data_json = json.load(f)
        self.data = [{'img_id': key.rsplit('_', 1)[0], 'img_id_v123': key, 'inpaint_img': key, 'score': value[0], 'rank': value[1]} for key, value in self.data_json.items()]
        
        self.get_fileinfo() 

        self.iters_per_epoch = int(math.ceil(len(self.data)*1.0/opts.batch_size))

    def get_fileinfo(self):
        inpaint_images_list = []
        scores_list = []
        ranks_list = []
        img_ids_list = []
        img_ids_v123_list = []
        
        for item in self.data:
            inpaint_images_list.append(item['inpaint_img'])
            scores_list.append(item['score'])
            ranks_list.append(item['rank'])
            img_ids_list.append(item['img_id'])
            img_ids_v123_list.append(item['img_id_v123'])
        self.inpaint_imgs = [os.path.join(self.root_path, 'inpaint', name) for name in inpaint_images_list]
        self.scores = scores_list
        self.ranks = ranks_list
        self.img_ids = img_ids_list 
        self.img_ids_v123 = img_ids_v123_list
        self.mask_imgs = [os.path.join(self.root_path, 'masks_color', name.rsplit('_', 1)[0]+'_mask_color') for name in inpaint_images_list]
        
    def get_file(self, index):
        inpaint = Image.open(self.inpaint_imgs[index] + '.jpg')
        mask_rgb = Image.open(self.mask_imgs[index] + '.jpg')
        inpaint = self.preprocess(inpaint).unsqueeze(0)
        mask_rgb = self.preprocess(mask_rgb).unsqueeze(0)
        score = torch.tensor([self.scores[index]], dtype=torch.float32)
        rank = torch.tensor([self.ranks[index]], dtype=torch.float32)
        img_id = self.img_ids[index]
        img_id_v123 = self.img_ids_v123[index]
        data = {
            'img_id': img_id,
            'img_id_v123': img_id_v123,
            "inpaint": inpaint,
            'mask_rgb': mask_rgb,
            'score': score,
            'rank': rank,
        }
        return data
        

    def __getitem__(self, index):
        return self.get_file(index)

    def __len__(self):
        return len(self.data)
    
    
class InpaintRewardDatasetGroup(Dataset):
    
    def __init__(self, dataset, root_path):

        self.better_inpaint_imgs = None
        self.better_mask_imgs = None
        self.worse_inpaint_imgs = None
        self.worse_mask_imgs = None
        self.scores = None
        self.root_path = root_path 
        
        self.clip, self.preprocess = clip.load("ViT-B/32", device="cuda")  
        self.dataset_path = os.path.join(config['data_base'], f"{dataset}.json")
        with open(self.dataset_path, "r") as f:
                self.data_json = json.load(f)
        self.data = [{'img_id': key, 'generations': value} for key, value in self.data_json.items()]
        
        self.get_fileinfo() 

        self.iters_per_epoch = int(math.ceil(len(self.data)*1.0/opts.batch_size))

    def get_fileinfo(self):
        better_inpaint_images_list = []
        worse_inpaint_images_list = []
        
        for id, generations in self.data_json.items():
            for img_name, value in generations.items():
                if value[1] == 1:  
                    better_inpaint_images_list.append(img_name)
                if value[1] == 3:
                    worse_inpaint_images_list.append(img_name)

        self.better_inpaint_imgs = [os.path.join(self.root_path, 'inpaint', name) for name in better_inpaint_images_list]
        self.worse_inpaint_imgs = [os.path.join(self.root_path, 'inpaint', name) for name in worse_inpaint_images_list]
        self.better_mask_imgs = [os.path.join(self.root_path, 'masks_color', name.rsplit('_', 1)[0]+'_mask_color') for name in better_inpaint_images_list]
        self.worse_mask_imgs = [os.path.join(self.root_path, 'masks_color', name.rsplit('_', 1)[0]+'_mask_color') for name in worse_inpaint_images_list]
    
    def get_file(self, index):

        better_inpaint = Image.open(self.better_inpaint_imgs[index] + '.jpg')
        better_mask = Image.open(self.better_mask_imgs[index] + '.jpg')
        better_inpaint = self.preprocess(better_inpaint).unsqueeze(0)
        better_mask = self.preprocess(better_mask).unsqueeze(0)

        worse_inpaint = Image.open(self.worse_inpaint_imgs[index] + '.jpg')
        worse_mask = Image.open(self.worse_mask_imgs[index] + '.jpg')
        worse_inpaint = self.preprocess(worse_inpaint).unsqueeze(0)
        worse_mask = self.preprocess(worse_mask).unsqueeze(0)

        data = {
            "better_inpt": better_inpaint,
            'better_msk': better_mask,
            "worse_inpt": worse_inpaint,
            'worse_msk': worse_mask,
        }

        return data
        

    def __getitem__(self, index):
        return self.get_file(index)

    def __len__(self):
        return len(self.data)

class InpaintRewardDatasetRegressionGroup(Dataset):
    
    def __init__(self, dataset, root_path):

        self.inpaint_imgs_1 = None
        self.mask_imgs_1 = None
        self.inpaint_imgs_2 = None
        self.mask_imgs_2 = None
        self.inpaint_imgs_3 = None
        self.mask_imgs_3 = None
        self.scores1 = None
        self.scores2 = None
        self.scores3 = None
        self.img_id = None
        self.root_path = root_path 
        
        self.clip, self.preprocess = clip.load("ViT-B/32", device="cuda")  
        self.dataset_path = os.path.join(config['data_base'], f"{dataset}.json")
        with open(self.dataset_path, "r") as f:
                self.data_json = json.load(f)
        self.data = [{'img_id': key, 'generations': value} for key, value in self.data_json.items()]
        
        self.get_fileinfo() 

        self.iters_per_epoch = int(math.ceil(len(self.data)*1.0/opts.batch_size))

    def get_fileinfo(self):
        inpaint_images_list_1 = []
        inpaint_images_list_1 = []
        inpaint_images_list_2 = []
        inpaint_images_list_2 = []
        inpaint_images_list_3 = []
        inpaint_images_list_3 = []
        score_1 = []
        score_2 = []
        score_3 = []
        img_id = []
        for id, generations in self.data_json.items():
            img_id.append(id)
            for img_name, value in generations.items():
                if value[1] == 1:
                    score_1.append(value[0])  
                    inpaint_images_list_1.append(img_name)
                elif value[1] == 2:
                    score_2.append(value[0])  
                    inpaint_images_list_2.append(img_name)
                else:
                    score_3.append(value[0])  
                    inpaint_images_list_3.append(img_name)


        self.inpaint_imgs_1 = [os.path.join(self.root_path, 'inpaint', name) for name in inpaint_images_list_1]
        self.inpaint_imgs_2 = [os.path.join(self.root_path, 'inpaint', name) for name in inpaint_images_list_2]
        self.inpaint_imgs_3 = [os.path.join(self.root_path, 'inpaint', name) for name in inpaint_images_list_3]
        self.mask_imgs_1 = [os.path.join(self.root_path, 'masks_color', name.rsplit('_', 1)[0]+'_mask_color') for name in inpaint_images_list_1]
        self.mask_imgs_2 = [os.path.join(self.root_path, 'masks_color', name.rsplit('_', 1)[0]+'_mask_color') for name in inpaint_images_list_2]
        self.mask_imgs_3 = [os.path.join(self.root_path, 'masks_color', name.rsplit('_', 1)[0]+'_mask_color') for name in inpaint_images_list_3]
        self.scores1 = score_1
        self.scores2 = score_2
        self.scores3 = score_3
        self.img_id = img_id
    
    def get_file(self, index):

        inpaint_1 = Image.open(self.inpaint_imgs_1[index] + '.jpg')
        mask_1 = Image.open(self.mask_imgs_1[index] + '.jpg')
        inpaint_1 = self.preprocess(inpaint_1).unsqueeze(0)
        mask_1 = self.preprocess(mask_1).unsqueeze(0)
        score_1 = torch.tensor(self.scores1[index])

        inpaint_2 = Image.open(self.inpaint_imgs_2[index] + '.jpg')
        mask_2 = Image.open(self.mask_imgs_2[index] + '.jpg')
        inpaint_2 = self.preprocess(inpaint_2).unsqueeze(0)
        mask_2 = self.preprocess(mask_2).unsqueeze(0)
        score_2 = torch.tensor(self.scores2[index])

        inpaint_3 = Image.open(self.inpaint_imgs_3[index] + '.jpg')
        mask_3 = Image.open(self.mask_imgs_3[index] + '.jpg')
        inpaint_3 = self.preprocess(inpaint_3).unsqueeze(0)
        mask_3 = self.preprocess(mask_3).unsqueeze(0)
        score_3 = torch.tensor(self.scores3[index])
        scores = torch.tensor([score_1, score_2, score_3])
        img_key = self.img_id[index]


        data = {
            "inpt_1": inpaint_1,
            'msk_1': mask_1,
            "inpt_2": inpaint_2,
            'msk_2': mask_2,
            "inpt_3": inpaint_3,
            'msk_3': mask_3,
            'score': scores,
            'img_key': img_key
        }

        return data
        

    def __getitem__(self, index):
        return self.get_file(index)

    def __len__(self):
        return len(self.data)
    