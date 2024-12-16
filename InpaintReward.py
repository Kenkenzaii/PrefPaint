import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision import transforms 


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
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
            # nn.Linear(128, 64),
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
        features = self.layers(input)
        out = self.last_layer(features)
        return out, features


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
        self.config = config
        self.device = device
        
        self.clip_model, self.preprocess = clip.load("ViT-B/32") #, device=self.device) #clip.load(config['clip_model'], device="cuda" if torch.cuda.is_available() else "cpu")
        
        self.clip_model = self.clip_model.float()
        self.mlp = MLP(self.config['ImageReward']['mlp_dim'])
        self.vit_block = ViTBlock(self.config["ViT"]["feature_dim"], self.config["ViT"]["num_heads"], self.config["ViT"]["mlp_dim"])
        
        self.toImage = transforms.ToPILImage()

        self.mean = 0.4064   #0.65823
        self.std =  2.3021       #8.5400

        if self.config.fix_base:
            self.clip_model.requires_grad_(False)
        
        # for name, parms in self.clip_model.named_parameters():
        #     if '_proj' in name:
        #         parms.requires_grad_(False)
        
        # # fix certain ratio of layers
        # self.image_layer_num = 12
        # if self.config.fix_rate > 0:
        #     image_fix_num = "resblocks.{}".format(int(self.image_layer_num * self.config.fix_rate))
        #     for name, parms in self.clip_model.visual.named_parameters():
        #         parms.requires_grad_(False)
        #         if image_fix_num in name:
        #             break

    
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
        emb_feature = self.vit_block(emb_feature) # 1024
      
        scores, last_features = self.mlp(emb_feature)
        scores = torch.squeeze(scores)
        last_features = torch.squeeze(last_features)

        if self.config.group:
            scores = (scores - self.mean) / self.std
    

        return scores.detach().cpu().numpy().tolist(), last_features.detach().cpu().numpy().tolist()
        

    
    def load_model(self, model, ckpt_path = None):
        
        print('load checkpoint from %s'%ckpt_path)
        state_dict = {k: v for k, v in torch.load(ckpt_path, map_location='cpu').items()}
        new_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        msg = model.load_state_dict(new_dict)
        # checkpoint = torch.load(ckpt_path, map_location='cpu') 
        # state_dict = checkpoint
        # msg = model.load_state_dict(state_dict,strict=False)
        
        return model 

       

class ImageRewardGroup(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda") #clip.load(config['clip_model'], device="cuda" if torch.cuda.is_available() else "cpu")
        
        self.clip_model = self.clip_model.float()
        self.mlp = MLP(config['ImageReward']['mlp_dim'])
        self.vit_block = ViTBlock(self.config["ViT"]["feature_dim"], self.config["ViT"]["num_heads"], self.config["ViT"]["mlp_dim"])
        
        if self.config.fix_base:
            self.clip_model.requires_grad_(False)
        
        for name, parms in self.clip_model.named_parameters():
            if '_proj' in name:
                parms.requires_grad_(False)
        
        # fix certain ratio of layers
        self.image_layer_num = 12
        if self.config.fix_base > 0:
            image_fix_num = "resblocks.{}".format(int(self.image_layer_num * self.config.fix_base))
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
        b_emb_feature = self.vit_block(b_emb_feature) # 1024
        w_emb_feature = torch.cat((w_emb_inpt, w_emb_msk), dim=-1)
        w_emb_feature = self.vit_block(w_emb_feature) # 1024

        reward_better = self.mlp(b_emb_feature).squeeze(-1)
        reward_worse = self.mlp(w_emb_feature).squeeze(-1)
        reward = torch.concat((reward_better, reward_worse), dim=1)
        
        return reward


    def encode_pair(self, batch_data):
        better_inpaint_embeds_bs, better_mask_rgb_embeds_bs = [], []
        worse_inpaint_embeds_bs, worse_mask_rgb_embeds_bs = [], []
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
    

    def load_model(self, model, ckpt_path = None):
        
        print('load checkpoint from %s'%ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location='cpu') 
        state_dict = checkpoint
        msg = model.load_state_dict(state_dict,strict=False)
        print("missing keys:", msg.missing_keys)

        return model 



