# Following instructions from notebook
# pip install -qq .
# pip install -qq tensorflow-gpu==2.1

import os
import numpy as np
import torch
import tensorflow as tf
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.saved_model import signature_constants, tag_constants
import yaml
from parallel_wavegan.models import MelGANGenerator

# setup pytorch model
vocoder_conf = './egs/ljspeech/voc1/conf/melgan.v1.long.yaml'
with open(vocoder_conf) as f:
    config = yaml.load(f, Loader=yaml.Loader)
pytorch_melgan = MelGANGenerator(**config["generator_params"])
pytorch_melgan.remove_weight_norm()
# TODO: Train MelGAN (Save state_dict to checkpoint and time to train)
pytorch_melgan = pytorch_melgan.to("cuda").eval()

# checks inference speed
fake_mels = np.random.sample((4, 1500, 80)).astype(np.float32)
with torch.no_grad():
    y = pytorch_melgan(fake_mels)

# TODO: check LJSpeech inference speed and MCD metric
