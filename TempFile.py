# temporary file for the sake of figuring out making batches
from collections import defaultdict
import os
# File to setup the image data to fine-tune the 
from collections import defaultdict
import os
import re
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import SegformerForSemanticSegmentation
from transformers import SegformerImageProcessor
import json
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from sklearn.metrics import accuracy_score
import evaluate
metric = evaluate.load("mean_iou")

# define the image processor
image_processor = SegformerImageProcessor(do_reduce_labels=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# customizable id's
road_labels = {
  "0": "not_road",
  "1": "road",
}
road_label2id = {v: k for k, v in road_labels.items()}


# load id2label mapping from a JSON on the hub
# think we might need to adjust the num_labels and the id2label situation to fit out specific case
repo_id = "huggingface/label-files"
filename = "ade20k-id2label.json"
id2label = json.load(open(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"), "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

def process_png(filename):
    image = Image.open(filename)
    binary_mask = np.array(image.convert("L"))
    binary_mask = np.where(binary_mask > 0, 1, 0)
    encoded_mask = np.zeros_like(binary_mask, dtype=np.int64)
    for label, class_name in road_labels.items():
        encoded_mask[binary_mask == label] = label
    return encoded_mask



path = '/Users/rorybeals/Downloads/Road Identification Data/train'
jpg_files = []
png_files = []
for file in os.listdir(path):
    if file.endswith('.jpg'):
        jpg_files.append(file)
    else: png_files.append(file)

# define our training data and labels
training_images = []
training_labels = []

data_dict = defaultdict(dict)
for file in os.listdir(path):
    num, type_ = file.split('_')
    # type_ = type_.split('.')[0]
    data_dict[num][type_] = file

print(data_dict['401242']['mask.png'])

for file in jpg_files:
    file_num = file.split('_')[0]
    filename = path + '/' + file
    training_images.append(filename)
    png_image = data_dict[file_num]['mask.png']
    png_filename = path + png_image
    training_labels.append(png_filename)

    sat_image = Image.open(filename)
    pixel_vals = image_processor(sat_image, return_tensors="pt").pixel_values.to(device)
    training_images.append(pixel_vals)
    png_image = data_dict[file_num]['mask.png']
    png_filename = path + '/' + png_image
    labels = process_png(png_filename)
    training_labels.append(labels)
    print(file)
    if (len(training_images) == 1000):
        break

tensor_list = [torch.tensor(arr) for arr in training_labels]


print(tensor_list[1])
print(type(tensor_list[1]))