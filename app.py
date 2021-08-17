import os
import random
import gradio as gr
from PIL import Image
import torch
from random import randint
import sys
from subprocess import call

torch.hub.download_url_to_file('https://i.imgur.com/tXrot31.jpg', 'cpu.jpg')


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

#run_cmd("python setup.py develop")


def inference(img):
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
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(INPUT_DIR + "1.jpg", "JPEG")
    run_cmd("python inference_realesrgan.py --model_path RealESRGAN_x4plus.pth --input "+ INPUT_DIR + " --output " + OUTPUT_DIR + " --netscale 4 --outscale 3.5")
    return os.path.join(OUTPUT_DIR, "1_out.jpg")

inferences_running = 0
def throttled_inference(image):
    global inferences_running
    current = inferences_running
    if current >= 3:
        print(f"Rejected inference when we already had {current} running")
        return "cpu.jpg"
    print(f"Inference starting when we already had {current} running")
    inferences_running += 1
    try:
        return inference(image)
    finally:
        print("Inference finished")
        inferences_running -= 1

        
title = "Real-ESRGAN"
description = "Gradio demo for Real-ESRGAN. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below. Please click submit only once"
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2107.10833'>Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data</a> | <a href='https://github.com/xinntao/Real-ESRGAN'>Github Repo</a></p>"

gr.Interface(
    throttled_inference, 
    [gr.inputs.Image(type="pil", label="Input")], 
    gr.outputs.Image(type="file", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=[
    ['bear.jpg']
    ]
).launch(debug=True)