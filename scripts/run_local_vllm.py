import argparse
import torch
import sys

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

import glob
import os.path

DELIMITER = "*****"
ACT_DELIMITER = "###"

model_path = "liuhaotian/llava-v1.5-7b"
model_base = None

def image_parser(image_file, sep):
    out = image_file.split(sep)
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def init_model(model_path, model_base, conv_mode=None):
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if conv_mode is not None and conv_mode != conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, conv_mode, conv_mode
            )
        )
    else:
        conv_mode = conv_mode
        
    return tokenizer, model, image_processor, conv_mode

def get_default_prompt(query, tokenizer, model, conv_mode, batch_size):
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    input_ids = torch.repeat_interleave(input_ids, repeats=batch_size, dim=0)
    return input_ids

def eval_batch(
    tokenizer, model, image_processor,
    input_ids, image_file_list
):
    images = load_images(image_file_list)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    
    temperature = 0
    top_p = None
    num_beams = 1
    max_new_tokens = 512
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [k.strip() for k in outputs]

def main():
    path = sys.argv[1]
    action_query = sys.argv[2]
    files = sorted(glob.glob(f"{path}/*.jpg"))
    sys.stderr.write("# files: " + str(len(files)) + "\n")
    sys.stderr.write(action_query + "\n")
    sys.stderr.flush()

    tokenizer, model, image_processor, conv_mode = init_model(model_path, model_base)

    with torch.no_grad():
        for filename in files:
            image_file_list = [filename]
            batch_size = 1
            action = action_query
            query = f"This is one frame from a longer video clip. The video clip includes the action described as '{action}' however, the frame may or may not include this action. Briefly explain what action or actions the person is conducting in the frame."
            input_ids = get_default_prompt(query, tokenizer, model, conv_mode, batch_size)
            outputs = eval_batch(tokenizer, model, image_processor, input_ids, image_file_list)
            filename_basename = os.path.basename(filename)[:-4]
            print(f"{DELIMITER} {filename_basename}{ACT_DELIMITER}{action} {DELIMITER}")
            print(outputs[0])
            sys.stdout.flush()

if __name__ == '__main__':
    main()
