import os, random
import json
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from transformers import BertTokenizer
import clip


class InpaintAlignmentDataset(Dataset):
    def __init__(self, config, dataset, root_path):

        self.config = config
        self.inpaint_imgs = None
        self.mask_imgs = None
        self.img_ids = None
        self.boundaries = None
        self.root_path = root_path
        self.transform = transforms.ToTensor()
        self.dataset_path = os.path.join(config["data_base"], f"{dataset}.txt")
        # with open(self.dataset_path, "r") as f:
            # self.data_json = json.load(f)
        
        with open(self.dataset_path, 'r') as txt_file:
            self.data_txt = [line.strip() for line in txt_file.readlines()]
       
        self.data = [
            {
                "img_id": line      
            }
            for line in self.data_txt
        ]
       
        self.get_fileinfo()

        self.iters_per_epoch = int(
            math.ceil(len(self.data) * 1.0 / config["batch_size"])
        )

    def get_fileinfo(self):
     
        img_ids_list = []
        # boundaries_list = []

        for item in self.data:
            # inpaint_images_list.append(item["inpaint_img"])
            # scores_list.append(item["score"])
            # ranks_list.append(item["rank"])
            img_ids_list.append(item["img_id"])
            # boundaries_list.append(item["boundary"])
        
           
        # self.inpaint_imgs = [
        #     os.path.join(self.root_path, "inpaint", name)
        #     for name in inpaint_images_list
        # ]
        self.masks = [
            os.path.join(self.root_path, "masks", name+ "_mask")
            for name in img_ids_list
        ]
        # self.scores = scores_list
        # self.ranks = ranks_list
        self.img_ids = img_ids_list
        # self.boundaries = boundaries_list
        self.mask_rgbs = [
            os.path.join(
                self.root_path, "masks_color", name + "_mask_color"
            )
            for name in img_ids_list
        ]

    def get_file(self, index):

        mask = Image.open(self.masks[index] + ".png").convert('L')
        mask_rgb = Image.open(self.mask_rgbs[index] + ".png")
        color, mask = np.array(mask_rgb).transpose(2, 0, 1), np.array(mask)
        mask = mask[None, ...]
            
        mask_ = np.zeros_like(mask)
        mask_[mask < 125] = 0
        mask_[mask >= 125] = 1
        color = (1 - mask_) * color

        # noise = np.random.normal(0, 255 * np.random.rand(), mask.shape)
        # color_noise = (1 - mask_) * color + mask_ * noise
                
        color = torch.from_numpy(color)
        mask = torch.from_numpy(mask)
        img_id = self.img_ids[index]
        # boundaries = torch.tensor(self.boundaries[index])

        # data = {
        #     "img_id": img_id,
        #     "inpaint": color,
        #     "mask_rgb": mask,
        # }
        return color, mask, img_id #, boundaries

    def __getitem__(self, index):
        return self.get_file(index)

    def __len__(self):
        return len(self.data)

class InpaintRewardDataset(Dataset):
    def __init__(self, config, dataset, root_path):

        self.config = config
        self.inpaint_imgs = None
        self.mask_imgs = None
        self.scores = None
        self.ranks = None
        self.root_path = root_path

        self.clip, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.dataset_path = os.path.join(config["data_base"], f"{dataset}.json")
        with open(self.dataset_path, "r") as f:
            self.data_json = json.load(f)
        self.data = [
            {
                "img_id": key.rsplit("_", 1)[0],
                "inpaint_img": key,
                "score": value[0],
                "rank": value[1],
            }
            for key, value in self.data_json.items()
        ]

        self.get_fileinfo()

        self.iters_per_epoch = int(
            math.ceil(len(self.data) * 1.0 / config["batch_size"])
        )

    def get_fileinfo(self):
        inpaint_images_list = []
        scores_list = []
        ranks_list = []
        img_ids_list = []

        for item in self.data:
            inpaint_images_list.append(item["inpaint_img"])
            scores_list.append(item["score"])
            ranks_list.append(item["rank"])
            img_ids_list.append(item["img_id"])
        self.inpaint_imgs = [
            os.path.join(self.root_path, "inpaint", name)
            for name in inpaint_images_list
        ]
        self.scores = scores_list
        self.ranks = ranks_list
        self.img_ids = img_ids_list
        self.mask_imgs = [
            os.path.join(
                self.root_path, "masks_color", name.rsplit("_", 1)[0] + "_mask_color"
            )
            for name in inpaint_images_list
        ]

    def get_file(self, index):

        inpaint = Image.open(self.inpaint_imgs[index] + ".png")
        mask_rgb = Image.open(self.mask_imgs[index] + ".png")
        inpaint = self.preprocess(inpaint).unsqueeze(0)
        mask_rgb = self.preprocess(mask_rgb).unsqueeze(0)
        score = torch.tensor([self.scores[index]], dtype=torch.float32)
        rank = torch.tensor([self.ranks[index]], dtype=torch.float32)
        img_id = self.img_ids[index]

        data = {
            "img_id": img_id,
            "inpaint": inpaint,
            "mask_rgb": mask_rgb,
            "score": score,
            "rank": rank,
        }
        return data

    def __getitem__(self, index):
        return self.get_file(index)

    def __len__(self):
        return len(self.data)


class InpaintRewardDatasetGroup(Dataset):
    def __init__(self, config, dataset, root_path):

        self.config = config
        self.better_inpaint_imgs = None
        self.better_mask_imgs = None
        self.worse_inpaint_imgs = None
        self.worse_mask_imgs = None
        self.scores = None
        self.root_path = (
            root_path  # '/data/kendong/Diffusions/joint-rl-diffusion/data/ade20k/merge'
        )

        self.clip, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.dataset_path = os.path.join(self.config["data_base"], f"{dataset}.json")
        with open(self.dataset_path, "r") as f:
            self.data_json = json.load(f)
        self.data = [
            {"img_id": key, "generations": value}
            for key, value in self.data_json.items()
        ]

        self.get_fileinfo()

        self.iters_per_epoch = int(
            math.ceil(len(self.data) * 1.0 / config["batch_size"])
        )

    def get_fileinfo(self):
        better_inpaint_images_list = []
        worse_inpaint_images_list = []

        for _, generations in self.data_json.items():
            for img_name, value in generations.items():
                if value[1] == 1:
                    better_inpaint_images_list.append(img_name)
                if value[1] == 3:
                    worse_inpaint_images_list.append(img_name)

        self.better_inpaint_imgs = [
            os.path.join(self.root_path, "inpaint", name)
            for name in better_inpaint_images_list
        ]
        self.worse_inpaint_imgs = [
            os.path.join(self.root_path, "inpaint", name)
            for name in worse_inpaint_images_list
        ]
        self.better_mask_imgs = [
            os.path.join(
                self.root_path, "masks_color", name.rsplit("_", 1)[0] + "_mask_color"
            )
            for name in better_inpaint_images_list
        ]
        self.worse_mask_imgs = [
            os.path.join(
                self.root_path, "masks_color", name.rsplit("_", 1)[0] + "_mask_color"
            )
            for name in worse_inpaint_images_list
        ]

    def get_file(self, index):

        better_inpaint = Image.open(self.better_inpaint_imgs[index] + ".png")
        better_mask = Image.open(self.better_mask_imgs[index] + ".png")
        better_inpaint = self.preprocess(better_inpaint).unsqueeze(0)
        better_mask = self.preprocess(better_mask).unsqueeze(0)

        worse_inpaint = Image.open(self.worse_inpaint_imgs[index] + ".png")
        worse_mask = Image.open(self.worse_mask_imgs[index] + ".png")
        worse_inpaint = self.preprocess(worse_inpaint).unsqueeze(0)
        worse_mask = self.preprocess(worse_mask).unsqueeze(0)

        data = {
            "better_inpt": better_inpaint,
            "better_msk": better_mask,
            "worse_inpt": worse_inpaint,
            "worse_msk": worse_mask,
        }

        return data

    def __getitem__(self, index):
        return self.get_file(index)

    def __len__(self):
        return len(self.data)
