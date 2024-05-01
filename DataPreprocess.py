# File to setup the image data to fine-tune the 
from collections import defaultdict
import os
import re
import cv2
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

# just kinda testing this out and figuring out if it works
image = Image.open("/Users/rorybeals/Downloads/Road Identification Data/test/206_sat.jpg")
png_image = Image.open("/Users/rorybeals/Downloads/Road Identification Data/train/562_mask.png")
model.to(device)
pixel_vals = image_processor(image, return_tensors="pt").pixel_values.to(device)
print(pixel_vals.shape)

# forward pass
with torch.no_grad():
  outputs = model(pixel_values=pixel_vals)

# took this from online to help with visualizing the segmentation
def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]
     
predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
print(predicted_segmentation_map.shape)

color_seg = np.zeros((predicted_segmentation_map.shape[0],
                      predicted_segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3

palette = np.array(ade_palette())
for label, color in enumerate(palette):
    color_seg[predicted_segmentation_map == label, :] = color
# Convert to BGR
color_seg = color_seg[..., ::-1]
# cv2.imwrite("image_out.jpg", color_seg)

# Show image + mask
img = np.array(image) * 0.5 + color_seg * 0.5
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()

print('ahhhhh')
# this function is supposed to process the png files and encode them with proper values that we can use to retrain the model a bit
def process_png(filename):
    image = Image.open(filename)
    binary_mask = np.array(image.convert("L"))
    binary_mask = np.where(binary_mask > 0, 1, 0)
    # encoded_mask = np.zeros_like(binary_mask, dtype=np.int64)
    # for label, class_name in road_labels.items():
    #     encoded_mask[binary_mask == label] = label
    return torch.tensor(binary_mask)

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
    if (len(training_images) == 20):
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

#adjust the loss function
import torch
import torch.nn as nn

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)

# Assuming class 1 (road) is more important than class 0 (not road)
weights = torch.tensor([1.0, 3.0])  # More weight for class 1

criterion = WeightedCrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)


# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
image_processor = SegformerImageProcessor(do_reduce_labels=False)
model.to(device)
# tune the model
print('hi6')






model.train()
for epoch in range(5):  # loop over the dataset multiple times
    print("Epoch:", epoch)
    for batch in dataloader:
        optimizer.zero_grad()
        pixel_values, labels = batch
        pixel_values = pixel_values[:, 0, :, :, :]
        pixel_values, labels = pixel_values.to(device), labels.to(device)

        outputs = model(pixel_values=pixel_values)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Additional evaluation or logging code here (e.g., calculate metrics)

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        # print('batch running')
        # # get the inputs;
        # pixel_values, labels = batch
        # pixel_values = pixel_values.to(device)
        # pixel_values = pixel_values[:, 0, :, :, :]
        # print(pixel_values[0].shape)
        # print(pixel_values[0])
        # print(pixel_values[1])
        # labels = labels.to(device)
        # # zero the parameter gradients
        # optimizer.zero_grad()
        # # forward + backward + optimize
        # outputs = model(pixel_values=pixel_values, labels=labels)
        # logits = outputs.logits

        # output1 = model(pixel_values=pixel_values[0].unsqueeze(0))
        # output2 = model(pixel_values=pixel_values[1].unsqueeze(0))

        # target_sizes = [image.size[::-1]]  # List of target sizes corresponding to each image in the batch
        # predicted_segmentation_map1 = image_processor.post_process_semantic_segmentation(output1, target_sizes=target_sizes)[0]
        # predicted_segmentation_map2 = image_processor.post_process_semantic_segmentation(output2, target_sizes=target_sizes)[0]
        # predicted_segmentation_map1 = tf.convert_to_tensor(predicted_segmentation_map1.cpu())
        # predicted_segmentation_map2 = tf.convert_to_tensor(predicted_segmentation_map2.cpu())
        # numpy_version1 = predicted_segmentation_map1.numpy()
        # numpy_version2 = predicted_segmentation_map2.numpy()
        # torch_map1 = torch.from_numpy(numpy_version1)
        # torch_map2 = torch.from_numpy(numpy_version2)
        # combined_tensor = torch.stack([torch_map1, torch_map2], dim=0)
        # print(combined_tensor.shape)

        # # torch_map = torch_map.to(torch.float32)
        # new_labels = labels.to(torch.float64)
        # #new_labels = new_labels.to(torch.long)  # Convert to long (torch.int64)

        # # print(type(torch_map))
        # # print(torch_map.shape)
        # # print(type(new_labels))
        # # print(new_labels.shape)
        # # reshaped_tensor = torch_map.view(2, 1024, 1024)
        # loss = criterion(combined_tensor, new_labels)

        # loss.backward()
        # optimizer.step()
        # print('gabagoo4')
        # # evaluate
        # with torch.no_grad():
        #   upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        #   predicted = upsampled_logits.argmax(dim=1)

        #   # note that the metric expects predictions + labels as numpy arrays
        #   metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
        #   metrics = metric._compute(
        #           predictions=predicted.cpu(),
        #           references=labels.cpu(),
        #           num_labels=len(id2label),
        #           ignore_index=255,
        #           reduce_labels=False, # we've already reduced the labels ourselves
        #       )
        print('batch over')

    # print("Loss:", loss.item())
    # print("Mean_iou:", metrics["mean_iou"])
    # print("Mean accuracy:", metrics["mean_accuracy"])

# just kinda testing this out and figuring out if it works
image = Image.open("/Users/rorybeals/Downloads/Road Identification Data/test/206_sat.jpg")
png_image = Image.open("/Users/rorybeals/Downloads/Road Identification Data/train/562_mask.png")
pixel_vals = image_processor(image, return_tensors="pt").pixel_values.to(device)

# forward pass
with torch.no_grad():
  outputs = model(pixel_values=pixel_vals)

     
predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
print(predicted_segmentation_map)

color_seg = np.zeros((predicted_segmentation_map.shape[0],
                      predicted_segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3

palette = np.array(ade_palette())
for label, color in enumerate(palette):
    color_seg[predicted_segmentation_map == label, :] = color
# Convert to BGR
color_seg = color_seg[..., ::-1]

# Show image + mask
img = np.array(image) * 0.5 + color_seg * 0.5
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()