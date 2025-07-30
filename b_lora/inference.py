import argparse

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

from typing import Optional

#CONTENT_LIST = ["black_teen", "indian_teen", "chinese_man", "dog", "lenna", "burocrat"]
CONTENT_LIST = ["idil"]
STYLE_LIST = ["cartoon", "drawing1","painting", "pen_sketch","watercolor"]

BLOCKS = {
    'content': ['unet.up_blocks.0.attentions.0'],
    'style': ['unet.up_blocks.0.attentions.1'],
}


def is_belong_to_blocks(key, blocks):
    try:
        for g in blocks:
            if g in key:
                return True
        return False
    except Exception as e:
        raise type(e)(f'failed to is_belong_to_block, due to: {e}')


def filter_lora(state_dict, blocks_):
    try:
        return {k: v for k, v in state_dict.items() if is_belong_to_blocks(k, blocks_)}
    except Exception as e:
        raise type(e)(f'failed to filter_lora, due to: {e}')


def scale_lora(state_dict, alpha):
    try:
        return {k: v * alpha for k, v in state_dict.items()}
    except Exception as e:
        raise type(e)(f'failed to scale_lora, due to: {e}')


def get_target_modules(unet, blocks=None):
    try:
        if not blocks:
            blocks = [('.').join(blk.split('.')[1:]) for blk in BLOCKS['content'] + BLOCKS['style']]

        attns = [attn_processor_name.rsplit('.', 1)[0] for attn_processor_name, _ in unet.attn_processors.items() if
                 is_belong_to_blocks(attn_processor_name, blocks)]

        target_modules = [f'{attn}.{mat}' for mat in ["to_k", "to_q", "to_v", "to_out.0"] for attn in attns]
        return target_modules
    except Exception as e:
        raise type(e)(f'failed to get_target_modules, due to: {e}')

def setup_args_for_inference(content, style):
    args = argparse.Namespace()
    args.prompt=f"A {content} in {style} style"
    args.output_path="/content/drive/MyDrive/c447_project_lora_weights/output/blora/style_transfer"
    args.content_B_LoRA=f"/content/drive/MyDrive/c447_project_lora_weights/blora/{content}/pytorch_lora_weights.safetensors"
    args.style_B_LoRA=f"/content/drive/MyDrive/c447_project_lora_weights/blora/{style}/pytorch_lora_weights.safetensors"
    args.content_alpha=1.0
    args.style_alpha=1.0
    args.num_images_per_prompt=4
    return args


if __name__ == '__main__':

  for content in CONTENT_LIST:
    for style in STYLE_LIST:
      args = setup_args_for_inference(content, style)
      vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
      pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                          vae=vae,
                                                          torch_dtype=torch.float16).to("cuda")

      # Get Content B-LoRA SD
      if args.content_B_LoRA is not None:
          content_B_LoRA_sd, _ = pipeline.lora_state_dict(args.content_B_LoRA)
          content_B_LoRA = filter_lora(content_B_LoRA_sd, BLOCKS['content'])
          content_B_LoRA = scale_lora(content_B_LoRA, args.content_alpha)
      else:
          content_B_LoRA = {}

      # Get Style B-LoRA SD
      if args.style_B_LoRA is not None:
          style_B_LoRA_sd, _ = pipeline.lora_state_dict(args.style_B_LoRA)
          style_B_LoRA = filter_lora(style_B_LoRA_sd, BLOCKS['style'])
          style_B_LoRA = scale_lora(style_B_LoRA, args.style_alpha)
      else:
          style_B_LoRA = {}

      # Merge B-LoRAs SD
      res_lora = {**content_B_LoRA, **style_B_LoRA}

      # Load
      pipeline.load_lora_into_unet(res_lora, None, pipeline.unet)

      # Generate
      images = pipeline(args.prompt, num_images_per_prompt=args.num_images_per_prompt).images

      # Save
      for i, img in enumerate(images):
          img.save(f'{args.output_path}/{args.prompt}_{i}.jpg')
