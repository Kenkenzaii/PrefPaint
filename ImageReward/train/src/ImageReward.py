import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
from config.options import *
from config.utils import *
from models.clip_pretrain import clip_pretrain
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPTextModel, CLIPTokenizer, logging, CLIPVisionModel, CLIPFeatureExtractor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        # CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class MLP(nn.Module):
    def __init__(self, input_size, return_feature=False):
        super().__init__()
        self.input_size = input_size
        self.return_feature = return_feature
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 512),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            # # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(256, 128)
            # # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(128, 64)
            # # nn.ReLU(),
            # nn.Linear(64, 1)
        )
        self.last_layer = nn.Linear(256, 1, bias=False)
        self.last_layer_weight = self.last_layer.weight
        
        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)

        for name, param in self.last_layer.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
        
        
    def forward(self, input):
        
        feature = self.layers(input)
        out = self.last_layer(feature)
        w = self.last_layer.weight
   
        return out, feature, w
  
class ViTBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, mlp_dim, dropout=0.1):
        super(ViTBlock, self).__init__()
        # Transformer encoder layer
        self.encoder_layer = TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True  # Input shape: (batch_size, seq_length, feature_dim)
        )
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x


class ImageReward(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.device = device
        
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda") #clip.load(config['clip_model'], device="cuda" if torch.cuda.is_available() else "cpu")
        
        self.clip_model = self.clip_model.float()
        self.mlp = MLP(config['ImageReward']['mlp_dim'])
        self.vit_block = ViTBlock(config["ViT"]["feature_dim"], config["ViT"]["num_heads"], config["ViT"]["mlp_dim"])
        self.mean = 0.65823 #0.4064
        self.std =  8.5400 #2.3021
        if opts.fix_base:
            self.clip_model.requires_grad_(False)
        
        for name, parms in self.clip_model.named_parameters():
            if '_proj' in name:
                parms.requires_grad_(False)
        
        # fix certain ratio of layers
        self.image_layer_num = 12
        if opts.fix_rate > 0:
            image_fix_num = "resblocks.{}".format(int(self.image_layer_num * opts.fix_rate))
            for name, parms in self.clip_model.visual.named_parameters():
                parms.requires_grad_(False)
                if image_fix_num in name:
                    break


    def loose_layer(self, fix_rate):
        text_layer_id = [f"layer.{id}" for id in range(int(12 * fix_rate), 13)]
        image_layer_id = [f"blocks.{id}" for id in range(int(24 * fix_rate), 25)]
        for name, parms in self.blip.text_encoder.named_parameters():
            for text_id in text_layer_id:
                if text_id in name:
                    parms.requires_grad_(True)
        for name, parms in self.blip.visual_encoder.named_parameters():
            for image_id in image_layer_id:
                if image_id in name:
                    parms.requires_grad_(True)

    def score(self, inpaint_list, masks_rgb):
        
        inpaint_embeds_bs, mask_rgb_embeds_bs = [], []
        for bs in range(len(inpaint_list)):
            if isinstance(inpaint_list[bs], torch.Tensor):
                inpaint = self.toImage(inpaint_list[bs])
            else:
                inpaint = inpaint_list[bs]
            inpaint = self.preprocess(inpaint).unsqueeze(0)
            if isinstance(masks_rgb[bs], torch.Tensor):
                mask_rgb = self.toImage(masks_rgb[bs])
            else:
                mask_rgb = masks_rgb[bs]
            mask_rgb = self.preprocess(masks_rgb[bs]).unsqueeze(0)
            inpt, msk = inpaint.to(self.device), mask_rgb.to(self.device)
            # with torch.no_grad():
            inpt_embeds = self.clip_model.encode_image(inpt).to(torch.float32)
            msk_embeds = self.clip_model.encode_image(msk).to(torch.float32)

            inpaint_embeds_bs.append(inpt_embeds.squeeze(0))
            mask_rgb_embeds_bs.append(msk_embeds.squeeze(0))


        emb_inpaint = torch.stack(inpaint_embeds_bs, dim=0)
        emb_mask_rgb = torch.stack(mask_rgb_embeds_bs, dim=0)
        emb_feature = torch.cat((emb_inpaint, emb_mask_rgb), dim=-1)
        emb_feature = emb_feature.unsqueeze(1)
        emb_feature = self.vit_block(emb_feature) 
        scores, features, _ = self.mlp(emb_feature)
    
        scores = scores.squeeze(0)
        features = features.squeeze(0)
   
        return scores, features
    
    def forward(self, batch_data):
        
        emb_inpaint, emb_mask_rgb = self.encode_pair(batch_data) 
        # forward
        emb_feature = torch.cat((emb_inpaint, emb_mask_rgb), dim=-1)
        emb_feature = emb_feature.unsqueeze(1)
        emb_feature = self.vit_block(emb_feature) # 1024

        score, feature, _ = self.mlp(emb_feature)
        score = score.squeeze(-1)
        # if config.group:
        score = (score - self.mean) / self.std
           
        return score, feature


    def encode_pair(self, batch_data):
        inpaint_embeds_bs, mask_rgb_embeds_bs = [], []
        # if not opts.distributed:
        #     for bs in range(len(batch_data)):
        #         inpaint, mask_rgb = batch_data[bs]['inpaint'], batch_data[bs]['mask_rgb']
        #         inpaint, mask_rgb = inpaint.to(self.device), mask_rgb.to(self.device)
        #         # with torch.no_grad():
        #         inpaint_embeds = self.clip_model.encode_image(inpaint).to(torch.float32)
        #         mask_rgb_embeds = self.clip_model.encode_image(mask_rgb).to(torch.float32)
        #         inpaint_embeds_bs.append(inpaint_embeds)
        #         mask_rgb_embeds_bs.append(mask_rgb_embeds)
        #     return torch.stack(inpaint_embeds_bs, dim=0), torch.stack(mask_rgb_embeds_bs, dim=0)
        # else:
        inpaint, mask_rgb = batch_data['inpaint'].squeeze(1) , batch_data['mask_rgb'].squeeze(1) 
        inpaint, mask_rgb = inpaint.to(self.device), mask_rgb.to(self.device)
        inpaint_embeds = self.clip_model.encode_image(inpaint).to(torch.float32)
        mask_rgb_embeds = self.clip_model.encode_image(mask_rgb).to(torch.float32)
        return inpaint_embeds, mask_rgb_embeds
        


class ImageRewardRegressionGroup(nn.Module):

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda") #clip.load(config['clip_model'], device="cuda" if torch.cuda.is_available() else "cpu")
        
        self.clip_model = self.clip_model.float()
        self.mlp = MLP(config['ImageReward']['mlp_dim'])
        self.vit_block = ViTBlock(config["ViT"]["feature_dim"], config["ViT"]["num_heads"], config["ViT"]["mlp_dim"])
        
        if opts.fix_base:
            self.clip_model.requires_grad_(False)
        
        for name, parms in self.clip_model.named_parameters():
            if '_proj' in name:
                parms.requires_grad_(False)
        
        # fix certain ratio of layers
        self.image_layer_num = 12
        if opts.fix_rate > 0:
            image_fix_num = "resblocks.{}".format(int(self.image_layer_num * opts.fix_rate))
            for name, parms in self.clip_model.visual.named_parameters():
                parms.requires_grad_(False)
                if image_fix_num in name:
                    break


    def loose_layer(self, fix_rate):
        text_layer_id = [f"layer.{id}" for id in range(int(12 * fix_rate), 13)]
        image_layer_id = [f"blocks.{id}" for id in range(int(24 * fix_rate), 25)]
        for name, parms in self.blip.text_encoder.named_parameters():
            for text_id in text_layer_id:
                if text_id in name:
                    parms.requires_grad_(True)
        for name, parms in self.blip.visual_encoder.named_parameters():
            for image_id in image_layer_id:
                if image_id in name:
                    parms.requires_grad_(True)


    def forward(self, batch_data):
        
        emb_inpt_1, emb_msk_1, emb_inpt_2, emb_msk_2, emb_inpt_3, emb_msk_3 = self.encode_pair(batch_data) # Nan
        # forward
        emb_feature_1 = torch.cat((emb_inpt_1, emb_msk_1), dim=-1)
        emb_feature_1 = emb_feature_1.unsqueeze(1)
        emb_feature_1 = self.vit_block(emb_feature_1) # 1024

        emb_feature_2 = torch.cat((emb_inpt_2, emb_msk_2), dim=-1)
        emb_feature_2 = emb_feature_2.unsqueeze(1)
        emb_feature_2 = self.vit_block(emb_feature_2) # 1024

        emb_feature_3 = torch.cat((emb_inpt_3, emb_msk_3), dim=-1)
        emb_feature_3 = emb_feature_3.unsqueeze(1)
        emb_feature_3 = self.vit_block(emb_feature_3) # 1024
        
 
        reward_1, _, _ = self.mlp(emb_feature_1)
        reward_2, _, _ = self.mlp(emb_feature_2)
        reward_3, _, _ = self.mlp(emb_feature_3)
        
        reward_1 = reward_1.squeeze(-1)
        reward_2 = reward_2.squeeze(-1)
        reward_3 = reward_3.squeeze(-1)
  
        reward = torch.concat((reward_1, reward_2, reward_3), dim=1)
        return reward, self.mlp.last_layer_weight


    def encode_pair(self, batch_data):

        inpt1, msk1 = batch_data['inpt_1'].squeeze(1), batch_data['msk_1'].squeeze(1)
        inpt1, msk1 = inpt1.to(self.device), msk1.to(self.device)
        inpt2, msk2 = batch_data['inpt_2'].squeeze(1), batch_data['msk_2'].squeeze(1)
        inpt2, msk2 = inpt2.to(self.device), msk2.to(self.device)
        inpt3, msk3 = batch_data['inpt_3'].squeeze(1), batch_data['msk_3'].squeeze(1)
        inpt3, msk3 = inpt3.to(self.device), msk3.to(self.device)

        inpaint_embeds1 = self.clip_model.encode_image(inpt1).to(torch.float32)
        mask_rgb_embeds1 = self.clip_model.encode_image(msk1).to(torch.float32)
        inpaint_embeds2 = self.clip_model.encode_image(inpt2).to(torch.float32)
        mask_rgb_embeds2 = self.clip_model.encode_image(msk2).to(torch.float32)
        inpaint_embeds3 = self.clip_model.encode_image(inpt3).to(torch.float32)
        mask_rgb_embeds3 = self.clip_model.encode_image(msk3).to(torch.float32)

        return inpaint_embeds1, mask_rgb_embeds1, inpaint_embeds2, mask_rgb_embeds2, inpaint_embeds3, mask_rgb_embeds3
        
        
class ImageRewardGroup(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        self.clip_model, self.preprocess = clip.load("ViT-B/32") #clip.load(config['clip_model'], device="cuda" if torch.cuda.is_available() else "cpu")
        
        self.clip_model = self.clip_model.float()
        self.mlp = MLP(config['ImageReward']['mlp_dim'])
        self.vit_block = ViTBlock(config["ViT"]["feature_dim"], config["ViT"]["num_heads"], config["ViT"]["mlp_dim"])
        
        if opts.fix_base:
            self.clip_model.requires_grad_(False)
        
        for name, parms in self.clip_model.named_parameters():
            if '_proj' in name:
                parms.requires_grad_(False)
        
        # fix certain ratio of layers
        self.image_layer_num = 12
        if opts.fix_rate > 0:
            image_fix_num = "resblocks.{}".format(int(self.image_layer_num * opts.fix_rate))
            for name, parms in self.clip_model.visual.named_parameters():
                parms.requires_grad_(False)
                if image_fix_num in name:
                    break


    def loose_layer(self, fix_rate):
        text_layer_id = [f"layer.{id}" for id in range(int(12 * fix_rate), 13)]
        image_layer_id = [f"blocks.{id}" for id in range(int(24 * fix_rate), 25)]
        for name, parms in self.blip.text_encoder.named_parameters():
            for text_id in text_layer_id:
                if text_id in name:
                    parms.requires_grad_(True)
        for name, parms in self.blip.visual_encoder.named_parameters():
            for image_id in image_layer_id:
                if image_id in name:
                    parms.requires_grad_(True)


    def forward(self, batch_data):
        
        b_emb_inpt, b_emb_msk, w_emb_inpt, w_emb_msk = self.encode_pair(batch_data) # Nan
        
        # forward
        b_emb_feature = torch.cat((b_emb_inpt, b_emb_msk), dim=-1)
        b_emb_feature = b_emb_feature.unsqueeze(1)
        b_emb_feature = self.vit_block(b_emb_feature) # 1024

        w_emb_feature = torch.cat((w_emb_inpt, w_emb_msk), dim=-1)
        w_emb_feature = w_emb_feature.unsqueeze(1)
        w_emb_feature = self.vit_block(w_emb_feature) # 1024
      
        reward_better, _, _ = self.mlp(b_emb_feature)
        reward_worse, _, _ = self.mlp(w_emb_feature)
        
        # if not opts.distributed:
        reward_better = reward_better.squeeze(-1)
        reward_worse = reward_worse.squeeze(-1)
        reward = torch.concat((reward_better, reward_worse), dim=1)
        
        return reward, self.mlp.last_layer_weight


    def encode_pair(self, batch_data):
        better_inpaint_embeds_bs, better_mask_rgb_embeds_bs = [], []
        worse_inpaint_embeds_bs, worse_mask_rgb_embeds_bs = [], []

        if not opts.distributed:

            for bs in range(len(batch_data)):
                better_inpt, better_msk = batch_data[bs]['better_inpt'], batch_data[bs]['better_msk']
                better_inpt, better_msk = better_inpt.to(self.device), better_msk.to(self.device)

                worse_inpt, worse_msk = batch_data[bs]['worse_inpt'], batch_data[bs]['worse_msk']
                worse_inpt, worse_msk = worse_inpt.to(self.device), worse_msk.to(self.device)
                # with torch.no_grad():
                better_inpaint_embeds = self.clip_model.encode_image(better_inpt).to(torch.float32)
                better_mask_rgb_embeds = self.clip_model.encode_image(better_msk).to(torch.float32)
                worse_inpaint_embeds = self.clip_model.encode_image(worse_inpt).to(torch.float32)
                worse_mask_rgb_embeds = self.clip_model.encode_image(worse_msk).to(torch.float32)

                better_inpaint_embeds_bs.append(better_inpaint_embeds)
                better_mask_rgb_embeds_bs.append(better_mask_rgb_embeds)
                worse_inpaint_embeds_bs.append(worse_inpaint_embeds)
                worse_mask_rgb_embeds_bs.append(worse_mask_rgb_embeds)

                b_inpt = torch.stack(better_inpaint_embeds_bs, dim=0)
                b_msk = torch.stack(better_mask_rgb_embeds_bs, dim=0)
                w_inpt = torch.stack(worse_inpaint_embeds_bs, dim=0)
                w_msk = torch.stack(worse_mask_rgb_embeds_bs, dim=0)
                return b_inpt, b_msk, w_inpt, w_msk
        
        else:
            
            better_inpt, better_msk = batch_data['better_inpt'].squeeze(1), batch_data['better_msk'].squeeze(1)
            better_inpt, better_msk = better_inpt.to(self.device), better_msk.to(self.device)
            worse_inpt, worse_msk = batch_data['worse_inpt'].squeeze(1), batch_data['worse_msk'].squeeze(1)
            worse_inpt, worse_msk = worse_inpt.to(self.device), worse_msk.to(self.device)

            better_inpaint_embeds = self.clip_model.encode_image(better_inpt).to(torch.float32)
            better_mask_rgb_embeds = self.clip_model.encode_image(better_msk).to(torch.float32)
            worse_inpaint_embeds = self.clip_model.encode_image(worse_inpt).to(torch.float32)
            worse_mask_rgb_embeds = self.clip_model.encode_image(worse_msk).to(torch.float32)

            return better_inpaint_embeds, better_mask_rgb_embeds, worse_inpaint_embeds, worse_mask_rgb_embeds
        
        
       
        

