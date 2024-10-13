
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import argparse
import os
import pandas as pd 
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Process image captioning.')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing images.')
    parser.add_argument('--max_new_tokens', type=int, default=500, help='Maximum number of new tokens to generate.')
    parser.add_argument('--fasttext_path', type=str, required=True, help='Path to the folder containing  ocr tokens.')
    parser.add_argument('--output_file', type=str, required=True, help='Output CSV file to save results.')
    args = parser.parse_args()

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        load_in_4bit=True)  
    model.to("cuda:0")

    image_files = [f for f in os.listdir(args.folder_path) if f.endswith(('.jpg', '.png'))]
    results = []

    for image_file in image_files:
        fasttext_path = os.path.join(args.fasttext_path, image_files.split('.')[0] + '.npy')
        texts = np.load(fasttext_path, allow_pickle=True)[()]['texts']
        texts = ', '.join(texts)
        
        image_path = os.path.join(args.folder_path, image_file)
        image = Image.open(image_path)

        conversation = [
            {

              "role": "user",
              "content": [
                  {"type": "text", "text": f"Hình ảnh chứa văn bản: {texts}. Hãy mô tả hình ảnh một cách chi tiết và câu mô tả cần chứa các dòng chữ xuất hiện trên các vật thể đã nêu ở trên, kết hợp chúng một cách tự nhiên với phần mô tả.'"},
                   {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

        result = processor.decode(output[0], skip_special_tokens=True)
        
        results.append({'image': image_file, 'caption': result})

    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()  