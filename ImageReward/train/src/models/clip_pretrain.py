'''
 * Adapted from BLIP (https://github.com/salesforce/BLIP)
'''

from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
import transformers
transformers.logging.set_verbosity_error()

import torch
import clip
from torch import nn
import torch.nn.functional as F

from models.blip import create_vit, init_tokenizer, load_checkpoint
from config.options import *
from config.utils import *

class CLIP_Pretrain(nn.Module):
    def __init__(self,    
                device,             
                clip_config = config['clip_model'],                    
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()

        self.model, self.preprocess = clip.load(clip_config, device=device)
        

        # text_width = self.text_encoder.config.hidden_size
        
        # self.vision_proj = nn.Linear(vision_width, embed_dim)
        # self.text_proj = nn.Linear(text_width, embed_dim)


def clip_pretrain(**kwargs):
    clip = CLIP_Pretrain(**kwargs)     
    return clip.model, clip.preprocess

