import os, time, random
os.environ['NCCL_P2P_DISABLE']='1' 
import torch
import yaml
import math
import numpy as np
import accelerate
from munch import munchify
from accelerate import Accelerator
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import torch.nn.functional as F_nn
from PIL import Image
from diffusers import UNet2DConditionModel
from diffusers import (
    StableDiffusionInpaintPipeline,
)

from InpaintReward import ImageReward
from InpaintRewardDataset import InpaintAlignmentDataset
from pipeline_sdinpt import StableDiffusionInpaintPipelineCustom
from ddim_with_prob import DDIMSchedulerCustom
from utilts import *




def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def lr_lambda(current_step: int):
    start_factor = config.lr_warmup_start_ratio or 0
    if current_step < config.lr_warmup_steps:
        return (1 - start_factor) * (
            float(current_step) / float(max(1.0, config.lr_warmup_steps))
        ) + start_factor
    return 1.0


def save_model_hook(models, weights, output_dir):
    if accelerator.is_main_process:
        for model in models:
            model.save_pretrained(os.path.join(output_dir, "unet"))
            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()


def load_model_hook(models, input_dir):
    for _ in range(len(models)):
        # pop models so that they are not loaded again
        model = models.pop()
        # load diffusers style into model
        load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
        model.register_to_config(**load_model.config)

        model.load_state_dict(load_model.state_dict())
        del load_model


def make_null_text(text_tokenizer, batch_size):
    uncond_prompt = text_tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=text_tokenizer.model_max_length,
        return_tensors="pt",
    )
    return uncond_prompt.input_ids


def make_image_prompt(config, pipeline, image, mask, dtype, device):
    mask_condition = pipeline.mask_processor.preprocess(
        mask,
        height=config.resolution,
        width=config.resolution,
        crops_coords=None,
        resize_mode="default",
    )
    init_image = pipeline.image_processor.preprocess(
        image,
        height=config.resolution,
        width=config.resolution,
        crops_coords=None,
        resize_mode="default",
    )
    init_image = init_image.to(dtype=torch.float32)
    masked_image = init_image * (mask_condition < 0.5)
    masked_image = masked_image.to(device=device, dtype=dtype)

    return masked_image, mask_condition, init_image


def make_mask_condition_input(config, mask_condition, vae_scale_factor, device, dtype):
    mask = torch.nn.functional.interpolate(
        mask_condition,
        size=(
            config.resolution // vae_scale_factor,
            config.resolution // vae_scale_factor,
        ),
    )
    mask = mask.to(device=device, dtype=dtype)
    mask = torch.cat([mask] * 2)
    return mask


def encode_masked_image(image_encoder, masked_image, scaling_factor):
    image = image_encoder(masked_image)
    image_latents = image.latent_dist.sample(generator=None)
    image_latents = scaling_factor * image_latents

    return image_latents


def refine_inpaint(mask, image, inpainted_image):
    refine_inpaint_imgs = []
    for i in range(len(inpainted_image)):
        mask_image_arr = np.array(mask[i].convert("L"))
        mask_image_arr = mask_image_arr[:, :, None]

        mask_image_arr = mask_image_arr.astype(np.float32) / 255.0
        mask_image_arr[mask_image_arr < 0.5] = 0
        mask_image_arr[mask_image_arr >= 0.5] = 1

        unmasked_unchanged_image_arr = (1 - mask_image_arr) * image[
            i
        ] + mask_image_arr * inpainted_image[i]
        unmasked_unchanged_image = Image.fromarray(
            unmasked_unchanged_image_arr.round().astype("uint8")
        )
        refine_inpaint_imgs.append(unmasked_unchanged_image)
    return refine_inpaint_imgs


def l1_loss(prev_sample_mean, prev_sample_mean_reference):
   
    loss = F_nn.l1_loss(prev_sample_mean, prev_sample_mean_reference, reduction='mean')

    return loss 

def print_to_file_and_console(text, file_path):
    with open(file_path, 'a') as f:
        print(text)
        f.write(text + '\n')


def load_config(path):
    with open(path) as file:
        config_dict = yaml.safe_load(file)
        config = munchify(config_dict)

    save_path = os.path.join(config.save_folder, config.exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    with open(f'{save_path}/{config.exp_name}.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    return config

config = load_config("./configs/config.yaml")
log_path = os.path.join(config.save_folder, config.exp_name, "{}.txt".format(config.exp_name))

writer = visualizer(config)

model = UNet2DConditionModel.from_pretrained(
    config.pretrained_model_name, subfolder="unet"  #, torch_dtype=torch.float16
)

model.enable_xformers_memory_efficient_attention()
model.enable_gradient_checkpointing()

accelerator = Accelerator(
    # gradient_accumulation_steps=config["gradient_accumulation_steps"],
    mixed_precision=config["mixed_precision"]
)

pipeline = StableDiffusionInpaintPipelineCustom.from_pretrained(
    config.pretrained_model_name,
    safety_checker=None,
    requires_safety_checker=False,
    # torch_dtype=torch.float16
)
text_encoder, vae, tokenizer, feature_extractor = (
    pipeline.text_encoder,
    pipeline.vae,
    pipeline.tokenizer,
    pipeline.feature_extractor,
)
text_encoder = text_encoder.to(accelerator.device)
text_encoder.requires_grad_(False)
vae = vae.to(accelerator.device)
vae.requires_grad_(False)
scaling_factor = vae.config.scaling_factor
pipeline.to(accelerator.device)

text_encoder = text_encoder.to(dtype=model.dtype)
vae = vae.to(dtype=model.dtype)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    betas=(config.adam_beta1, config.adam_beta2),
    weight_decay=config.weight_decay,
    eps=config.adam_epsilon,
)

# optimizer_state_dict = torch.load(save_path_optm,  map_location='cpu')
# optimizer.load_state_dict(optimizer_state_dict)
lr_scheduler = LambdaLR(optimizer, lr_lambda)

train_dataset = InpaintAlignmentDataset(config, config.training_list, config["root_path"])
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

accelerator.register_save_state_pre_hook(save_model_hook)
accelerator.register_load_state_pre_hook(load_model_hook)

train_iter = iter(train_dataloader)

model.train()

pipeline.unet.to(dtype=accelerator.unwrap_model(model).dtype)
RM = ImageReward(config, accelerator.device).to(accelerator.device)
reward_model = RM.load_model(
    RM, config.ImageReward.model_path
) 
reward_model = reward_model.to(accelerator.device)
reward_model.requires_grad_(False)

V_inverse = torch.load(config.V_inverse_path, map_location='cpu') 
V_inverse = V_inverse.to(accelerator.device)

pipeline.unet.requires_grad_(False)
pipeline.safety_checker = None
pipeline.scheduler = DDIMSchedulerCustom.from_config(pipeline.scheduler.config)
pipeline.scheduler.set_timesteps(
    num_inference_steps=config.n_inference_steps, device=accelerator.device
)
tensor2Image = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(
            (config.resolution, config.resolution),
            interpolation=F.InterpolationMode.LANCZOS,
        ),
    ]
)

cnt = 0
current_step = 0
num_itertaions = 0
stack = MovingAverageStack(config.queue_length)


while config.max_iterations == 0 or num_itertaions < config.max_iterations:
    # with accelerator.accumulate(model):
        
        start_time = time.time()
        try:
            image, mask, img_id = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            image, mask, img_id = next(train_iter)
        image = [tensor2Image(im) for im in image]
        mask = [tensor2Image(msk) for msk in mask]
      
        data_load_time = time.time() - start_time
        num_itertaions += 1
        start_time = time.time()
        
        with torch.no_grad():
            prompt_tokens, negative_prompt_tokens = (
                make_null_text(tokenizer, config.batch_size),
                make_null_text(tokenizer, config.batch_size),
            )
            prompt_embeds, negative_prompt_embeds = (
                text_encoder(prompt_tokens.to(accelerator.device))[0],
                text_encoder(negative_prompt_tokens.to(accelerator.device))[0],
            )

            masked_image, mask_condition, init_image = make_image_prompt(
                config, pipeline, image, mask, prompt_embeds.dtype, accelerator.device
            )
            masked_image_latents = encode_masked_image(
                vae.encode, masked_image, scaling_factor
            )

        if (num_itertaions) % config.rl_model_update == 0 and num_itertaions != 0:
            pipeline.unet.load_state_dict(accelerator.unwrap_model(model).state_dict())
         

        with torch.no_grad():
            (
                final_latents,
                latents_lst,
                next_latents_lst,
                ts_lst,
                prev_sample_mean_lst,
                log_probs_mean_lst,
                pred_img_t_ori,
            ) = pipeline(
                image=init_image,
                mask_condition=mask_condition,
                masked_image_latents=masked_image_latents,
                eta=config.eta,
                prompt_embeds=prompt_embeds,  
                negative_prompt_embeds=negative_prompt_embeds,
            )

            reward_scores, last_features = reward_model.score(
                pred_img_t_ori, image
            ) 
            reward_scores_ori = reward_scores

            if isinstance(reward_scores, list):
                reward_scores = torch.Tensor(reward_scores).to(accelerator.device)
                last_features = torch.Tensor(last_features).to(accelerator.device)

            all_rewards = accelerator.gather(reward_scores)
            all_rewards_valid = all_rewards[~torch.isnan(all_rewards)]
            
           
            advantages = (reward_scores - torch.mean(all_rewards_valid)) / (
                                torch.std(all_rewards_valid) + 1e-7)
         
            advantages = torch.clamp(advantages, -config.ADV_CLIP_MAX, config.ADV_CLIP_MAX)
       
 
        ddpo_ratio_vals_lst, ddpo_loss_vals_lst, mse_loss_vals_lst = [], [], []
        loop_train_steps = config.n_train_steps
        cnt = 0
        for i in random.sample(range(config.n_inference_steps), k=loop_train_steps):  # latents_lst.shape: torch.Size([6, 10, 4, 64, 64])
            
            latents_i, next_latents_i, t_i = (
                                            latents_lst[:, i],
                                            next_latents_lst[:, i],
                                            ts_lst[:, i],
                                            )
            # pre-processing of model input
            latent_model_input = pipeline.scheduler.scale_model_input(
                torch.cat([latents_i] * 2), t_i
            )
            mask_input = make_mask_condition_input(
                config,
                mask_condition,
                pipeline.vae_scale_factor,
                accelerator.device,
                prompt_embeds.dtype,
            )
            masked_image_latents_input = torch.cat([masked_image_latents] * 2)
            masked_image_latents_input = masked_image_latents_input.to(
                accelerator.device, prompt_embeds.dtype
            )
            latent_model_input = torch.cat(
                [latent_model_input, mask_input, masked_image_latents_input], dim=1
            )
            prompt_embeds_input = torch.cat([prompt_embeds, negative_prompt_embeds])
            t_input = torch.cat([t_i] * 2)

            noise_pred_unet = model(
                latent_model_input, t_input, encoder_hidden_states=prompt_embeds_input
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred_unet.chunk(2)
            noise_pred_ddpo = noise_pred_uncond + config.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            l, _, prev_sample_mean, log_prob_mean = pipeline.scheduler.step(
                noise_pred_ddpo,
                t_i[0],
                latents_i,
                eta=config.eta,
                prev_sample=next_latents_i,
                return_dict=False
            )
            
            L1_loss = l1_loss(prev_sample_mean, prev_sample_mean_lst[:, i])
            ratio_ddpo = torch.exp(log_prob_mean - log_probs_mean_lst[:, i])
    
            unclipped_loss_ddpo = -advantages * ratio_ddpo
            clipped_loss_ddpo = -advantages * torch.clamp(
                        ratio_ddpo, 1.0 - config.clip_range, 1.0 + config.clip_range
                    )

            
            boundary = torch.sqrt(torch.sum(torch.pow(torch.matmul(last_features, V_inverse), 2), dim=1))
            # print_to_file_and_console(f'boundary shape and device {boundary, boundary.shape, boundary.device, torch.max(boundary), torch.min(boundary)}', log_path)
            boundary_weight = torch.exp(-config.boundary_alpha * boundary + config.boundary_beta)
            # print_to_file_and_console(f'boundary weight shape and device {boundary_weight, boundary_weight.shape, boundary.device}', log_path)
            boundary_weight = torch.clamp(boundary_weight, min=1.0, max=2.21)
            # print_to_file_and_console(f'boundary weight shape and device {boundary_weight, boundary.device}', log_path)
            
           
            loss_ddpo = boundary_weight * torch.max(unclipped_loss_ddpo, clipped_loss_ddpo)
            loss_ddpo = torch.sum(loss_ddpo)
            total_loss =  config.ddpo_loss_scale * loss_ddpo


            # Backpropagate
            accelerator.backward(total_loss)
            
            if num_itertaions % config.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


        # Gather all losses from all processes
        all_losses = accelerator.gather(total_loss)
        all_scores = accelerator.gather(loss_ddpo)
        all_reward_ori = accelerator.gather(torch.tensor(reward_scores_ori).to(accelerator.device))
        all_l1_losses = accelerator.gather(L1_loss)
            
        if accelerator.is_main_process:
            mean_loss = sum(all_losses) / (accelerator.num_processes * config.batch_size)
            mean_all_scores = sum(all_scores) / (accelerator.num_processes  * config.batch_size)
            mean_l1 = sum(all_l1_losses) / (accelerator.num_processes * config.batch_size)
            mean_reward_ori = sum(all_reward_ori) / (accelerator.num_processes * config.batch_size)
            writer.add_scalar('Reward-Loss', mean_all_scores, global_step=num_itertaions)
            writer.add_scalar('Reward-ori-Loss', mean_reward_ori, global_step=num_itertaions)
            writer.add_scalar('Total-Loss', mean_loss, global_step=num_itertaions)
            writer.add_scalar('L1-Loss', mean_l1, global_step=num_itertaions)
            writer.add_scalar('Gathered reward mean', torch.mean(all_rewards), global_step=num_itertaions)

        iteration_time = time.time() - start_time
        print_to_file_and_console(f'iteration time: {iteration_time},  img_id: {img_id} ', log_path)
     
        


        if accelerator.sync_gradients:
            if (
                (num_itertaions) % config.save_interval == 0
                or num_itertaions == config.max_iterations
            ) and num_itertaions != 0:
                accelerator.wait_for_everyone()
                #  Create the pipeline using the trained modules and save it.
                if accelerator.is_main_process:
                    # save_training_state(accelerator, current_step, config)
                    unet_save = accelerator.unwrap_model(model)
                    pipeline_save = StableDiffusionInpaintPipeline.from_pretrained(
                        config.pretrained_model_name,
                        text_encoder=text_encoder,
                        vae=vae,
                        unet=unet_save,
                        scheduler=pipeline.scheduler,
                    )
                    save_path = os.path.join(
                        config.save_folder, config.exp_name, f"iteration_{num_itertaions}"
                    )
                    save_path_optm = os.path.join(
                        config.save_folder, config.exp_name, f"iteration_{num_itertaions}/optimizer.pt"
                    )
                    pipeline_save.save_pretrained(save_path)
                    # optimizer_state_dict = optimizer.state_dict()
                    # torch.save(optimizer_state_dict, save_path_optm)
                    accelerator.print(f"Pipeline saved to {save_path}")
                    # print_to_file_and_console(f"Pipeline saved to {save_path}", log_path)
            # current_step += 1
            log_str = (
                f"iteration num is {num_itertaions}, device is{accelerator.device}"
            )
            # accelerator.print(f"{log_str}")
            # print_to_file_and_console(f"{log_str}", log_path)
