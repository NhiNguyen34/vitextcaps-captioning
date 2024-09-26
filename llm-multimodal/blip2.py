# !pip install --upgrade -q transformers bitsandbytes datasets accelerate
# !pip install git+https://github.com/huggingface/transformers.git

import os
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import argparse

def load_model():
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True,
                                                          torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

def generate_captions(folder_path, processor, model, device, prompt="Image of", max_new_tokens=415):
    captions = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('RGB')

            inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            captions.append({'filename': filename, 'caption': generated_text})
    return captions

def main(folder_path, output_file, max_new_tokens):
    processor, model, device = load_model()
    captions = generate_captions(folder_path, processor, model, device, max_new_tokens=max_new_tokens)
    df = pd.DataFrame(captions)
    df.to_csv(output_file, index=False)  # Save to output file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Captioning with BLIP-2")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing images")
    parser.add_argument("output_file", type=str, help="Output CSV file for captions")
    parser.add_argument("max_new_tokens", type=int, help="max_new_tokens")
    args = parser.parse_args()

    main(args.folder_path, args.output_file, args.max_new_tokens)
