# BLIP-2 Image Captioning

This repository provides a script to generate captions for images using the BLIP-2 model.

## Requirements

Make sure you have the following packages installed:
bash
pip install --upgrade -q transformers bitsandbytes datasets accelerate
pip install git+https://github.com/huggingface/transformers.git


## Usage

1. **Prepare your images**: Place all images you want to caption in a single folder. Supported formats are `.jpg`, `.jpeg`, and `.png`.

2. **Run the script**: Use the following command to run the script, replacing `<folder_path>`, `<output_file>`, and `<max_new_tokens>` with your desired values.

bash
python llm-multimodal/blip2.py <folder_path> <output_file> <max_new_tokens>


- `<folder_path>`: Path to the folder containing images.
- `<output_file>`: Name of the output CSV file where captions will be saved.
- `<max_new_tokens>`: Maximum number of new tokens to generate for each caption.

## Example

bash
python llm-multimodal/blip2.py ./images captions.csv 415


This command will generate captions for all images in the `./images` folder and save them to `captions.csv`.

## Notes

- Ensure you have a compatible GPU for optimal performance, as the model is loaded with 4-bit precision.
- Adjust the `max_new_tokens` parameter based on your needs for longer or shorter captions.