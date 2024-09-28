%cd /content
!git clone -b v1.0 https://github.com/camenduru/LLaVA
%cd /content/LLaVA

!pip install -q transformers==4.36.2
!pip install -q gradio .


import os
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
import pandas as pd
from tqdm import tqdm

model_path = "4bit/llava-v1.5-13b-3GB"
kwargs = {"device_map": "auto"}
kwargs['load_in_4bit'] = True
kwargs['quantization_config'] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# Load vision tower for processing images
vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device='cuda')
image_processor = vision_tower.image_processor

# Function to generate a caption for an image based on a prompt
def caption_image(image_file, prompt):
    # Load image from file or URL
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')

    # Prepare the image for model input
    disable_torch_init()
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    # Prepare the input prompt
    inp = f"{roles[0]}: {prompt}"
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()

    # Tokenize the input
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # Generate output
    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2,
                                    max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])

    # Decode and return the output
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    output = outputs.rsplit('</s>', 1)[0]
    return image, output

def automate_captioning(image_dir, instruction_file):
    # Load instructions from the file (one instruction per line)
    instructions_df = pd.read_csv(instruction_file)

    # Check if the 'instructions' column exists
    # If not, it might be named differently, e.g., 'instruction'
    # Adjust the column name below accordingly
    if 'instructions' not in instructions_df.columns:
        # Replace 'instruction' with the actual column name in your CSV file
        instruction_column_name = 'instruction'
    else:
        instruction_column_name = 'instructions'

    # List all image files in the directory
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    # Loop through each image
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"\nProcessing Image: {image_file}")

        # Loop through each instruction and apply it to the current image
        for index, row in instructions_df.iterrows():
            # Access the instruction using the correct column name
            instruction = row[instruction_column_name]
            image, output = caption_image(image_path, instruction.strip())
            print(f"Instruction: {instruction.strip()}, Output: {output}\n")



from google.colab import drive


# Step 1: Mount Google Drive
drive.mount('/content/drive')


# Example usage
image_dataset_path = '/content/drive/MyDrive/image_dataset'  # Path to your image dataset
instructions_file_path = '/content/intructions.csv'  # Path to your instructions file (CSV format)



automate_captioning(image_dataset_path, instructions_file_path)
