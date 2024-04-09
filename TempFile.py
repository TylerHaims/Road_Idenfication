# temporary file for the sake of figuring out making batches
from collections import defaultdict
import os


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

print(data_dict['438785']['mask.png'])
data_dict.keys()  # amount of data point