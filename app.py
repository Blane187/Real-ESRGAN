import os
os.system("pip install gradio==2.9b23")
import random
import gradio as gr
from PIL import Image
import torch
from random import randint
import sys
from subprocess import call
import psutil




torch.hub.download_url_to_file('http://people.csail.mit.edu/billf/project%20pages/sresCode/Markov%20Random%20Fields%20for%20Super-Resolution_files/100075_lowres.jpg', 'bear.jpg')
  
    
def run_cmd(command):
    try:
        print(command)
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)
run_cmd("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P .")
run_cmd("pip install basicsr")
run_cmd("pip freeze")

os.system("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P .")


def inference(img,mode):
    _id = randint(1, 10000)
    INPUT_DIR = "/tmp/input_image" + str(_id) + "/"
    OUTPUT_DIR = "/tmp/output_image" + str(_id) + "/"
    run_cmd("rm -rf " + INPUT_DIR)
    run_cmd("rm -rf " + OUTPUT_DIR)
    run_cmd("mkdir " + INPUT_DIR)
    run_cmd("mkdir " + OUTPUT_DIR)
    basewidth = 256
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.LANCZOS)
    img.save(INPUT_DIR + "1.jpg", "JPEG")
    if mode == "base":
        run_cmd("python inference_realesrgan.py -n RealESRGAN_x4plus -i "+ INPUT_DIR + " -o " + OUTPUT_DIR)
    else:
        os.system("python inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i "+ INPUT_DIR + " -o " + OUTPUT_DIR)
    return os.path.join(OUTPUT_DIR, "1_out.jpg")



        

inputs = [
    gr.inputs.Image(type="pil", label="Upload Image"),
    gr.inputs.Radio(["base", "anime"], label="Mode")
]
output = gr.outputs.Image(type="pil", label="Enhanced Image")

title = "Real-ESRGAN Demo"
description = "Enhance your images using Real-ESRGAN."
examples = [["example_image.jpg", "base"], ["example_image.jpg", "anime"]]

theme = 'Hev832/soft'

gr.Interface(inference, inputs, output, title=title, description=description, examples=examples, theme=theme).launch()
