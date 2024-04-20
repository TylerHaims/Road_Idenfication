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
from torch.utils.data import TensorDataset, DataLoader

print('hi')

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

print('hi2')
# define model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                         num_labels=2,
                                                         id2label=road_labels,
                                                         label2id=road_label2id,
)

print('hi3')
# define the image processor
image_processor = SegformerImageProcessor(do_reduce_labels=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# this function is supposed to process the png files and encode them with proper values that we can use to retrain the model a bit
def process_png(filename):
    image = Image.open(filename)
    binary_mask = np.array(image.convert("L"))
    binary_mask = np.where(binary_mask > 0, 1, 0)
    encoded_mask = np.zeros_like(binary_mask, dtype=np.int64)
    for label, class_name in road_labels.items():
        encoded_mask[binary_mask == label] = label
    return torch.tensor(encoded_mask)

# gonna see if we can train this thangggg
# get all of the filenames separated
path = '/Users/rorybeals/Downloads/Road Identification Data/train/'
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
    data_dict[num][type_] = file

# set up the training_images and training_labels lists of filenames
for file in jpg_files:
    file_num = file.split('_')[0]
    filename = path + '/' + file
    training_images.append(filename)
    png_image = data_dict[file_num]['mask.png']
    png_filename = path + '/'+  png_image
    training_labels.append(png_filename)
    if (len(training_images) == 1000):
        break

processed_images = []
processed_labels = []

for image_filename, label_filename in zip(training_images, training_labels):
    sat_image = Image.open(image_filename)
    pixel_vals = image_processor(sat_image, return_tensors="pt").pixel_values.to(device)
    processed_images.append(pixel_vals)
    labels = process_png(png_filename)
    processed_labels.append(labels)

print('hi4')

# Convert lists of tensors to PyTorch tensors
processed_images_tensor = torch.stack(processed_images)
processed_labels_tensor = torch.stack(processed_labels)

dataset = TensorDataset(processed_images_tensor, processed_labels_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# set up some batches
training_images_tensor = tf.convert_to_tensor(training_images)
training_labels_tensor = tf.convert_to_tensor(training_labels)

print('hi5')

# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
image_processor = SegformerImageProcessor(do_reduce_labels=True)
model.to(device)
# tune the model
print('hi6')
model.train()
for epoch in range(200):  # loop over the dataset multiple times
    print("Epoch:", epoch)
    for batch in dataloader:
        # get the inputs;
        pixel_values, labels = batch
        pixel_values = pixel_values.to(device)
        pixel_values = pixel_values[:, 0, :, :, :]
        labels = labels.to(device)
        print('gabagoo')
        # zero the parameter gradients
        optimizer.zero_grad()
        print('gabagoo2')
        # forward + backward + optimize
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits
        print('gabagoo3')

        loss.backward()
        optimizer.step()
        print('gabagoo4')
        # evaluate
        with torch.no_grad():
          upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
          predicted = upsampled_logits.argmax(dim=1)

          # note that the metric expects predictions + labels as numpy arrays
          metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
          metrics = metric._compute(
                  predictions=predicted.cpu(),
                  references=labels.cpu(),
                  num_labels=len(id2label),
                  ignore_index=255,
                  reduce_labels=False, # we've already reduced the labels ourselves
              )

    print("Loss:", loss.item())
    print("Mean_iou:", metrics["mean_iou"])
    print("Mean accuracy:", metrics["mean_accuracy"])