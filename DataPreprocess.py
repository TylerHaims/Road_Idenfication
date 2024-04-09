import os
import re
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

# define model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                         num_labels=2,
                                                         id2label=road_labels,
                                                         label2id=road_label2id,
)

# this function is supposed to process the png files and encode them with proper values that we can use to retrain the model a bit
def process_png(filename):
    image = Image.open(filename)
    binary_mask = np.array(image.convert("L"))
    binary_mask = np.where(binary_mask > 0, 1, 0)
    encoded_mask = np.zeros_like(binary_mask, dtype=np.int64)
    for label, class_name in road_labels.items():
        encoded_mask[binary_mask == label] = label
    return encoded_mask

# gonna see if we can train this thangggg
# get all of the filenames separated
path = '/Users/rorybeals/Downloads/Road Identification Data/train'
jpg_files = []
png_files = []
for file in os.listdir(path):
    if file.endswith('.jpg'):
        jpg_files.append(file)
    else: png_files.append(file)

# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
image_processor = SegformerImageProcessor(do_reduce_labels=True)
# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# tune the model
model.train()
for file in jpg_files:
    # find the proper .png file
    match = re.match(r'^(\d+)', file)
    id = match.group()
    for fil in png_files:
        new_match = re.match(r'^(\d+)', fil)
        new_id = new_match.group()
        if id == new_id:
            png_file = fil
    file = "/Users/rorybeals/Downloads/Road Identification Data/train/" + file
    png_file = "/Users/rorybeals/Downloads/Road Identification Data/train/" + png_file
    image = Image.open(file)
    pixel_vals = image_processor(image, return_tensors="pt").pixel_values.to(device)
    labels = process_png(png_file)

    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = model(pixel_values=pixel_vals, labels=torch.tensor(labels))
    loss, logits = outputs.loss, outputs.logits

    loss.backward()
    optimizer.step()

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