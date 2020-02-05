#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess Celeb A dataset
"""

wd = "/home/xin/Documents/celeba_preprocessing"

import os
os.chdir(wd)
import pickle
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
import random
import csv
import itertools
import gc


#import torch
#import torchvision
#import torchvision.transforms as transforms
#from torchvision.utils import save_image

#sel_attr_names = ["Male", "Smiling", "Attractive"]
sel_attr_names = ["Wearing_Lipstick", "Smiling", "Mouth_Slightly_Open"]
#sel_attr_names = ["Male", "Smiling"]

IMG_SIZE = 64
NC = 3
N_CLASS = 2**len(sel_attr_names)
Resized_Method = "BILINEAR"; #"BILINEAR" or "LANCZOS"

np.random.seed(2019)
random.seed(2019)

###############################################################################
# 1. Load attributes
path_attr = wd + "/list_attr_celeba.txt"
attr_data = np.loadtxt(path_attr, skiprows=2, usecols=range(1,41), dtype=np.int)
image_filename = []
with open(path_attr, 'r') as infile:
    cnt = 1
    for line in infile:
        if cnt == 2: # skip the first row and only read the second row; first row is a number 202599
            attr_name = line
        if cnt > 2:
            line = line.split(" ")
            image_filename.append(line[0])
        cnt+=1
attr_name = attr_name.split(" ")
attr_name.pop()
assert len(attr_name) == 40
print(attr_data.shape)
print(len(image_filename))
assert len(image_filename) == attr_data.shape[0]
assert attr_data.shape[1] == 40
# convert -1 in attr_data to 0
attr_data[attr_data==-1]=0

# attribute name to column index
attr_to_idx = {}
for i in range(40):
    attr_to_idx[attr_name[i]] = i

# dump attribute names and name_to_indx dict into pickle file
attr_info = {"attr_name": attr_name, "name2idx": attr_to_idx}
with open("attr_info.pkl", "wb") as dump_file:
    pickle.dump(attr_info, dump_file)

# indices of selected attributes
sel_indices = []
for sel_attr in sel_attr_names:
    sel_indices.append(attr_to_idx[sel_attr])
    print(sel_attr+", Positive " + str(sum(attr_data[:,attr_to_idx[sel_attr]]==1)) + ", Negative " + str(sum(attr_data[:,attr_to_idx[sel_attr]]==0)))



# assign a label to each combination.
attr_comb_to_label = {}
all_combinations = list(itertools.product([0, 1], repeat=len(sel_attr_names)))
for i in range(len(all_combinations)):
    attr_comb_to_label[all_combinations[i]] = i 

###############################################################################
# 2. Load images
h5py_all_images_file = './celeba_64x64_only_images.h5'

N_all = attr_data.shape[0]
if not os.path.isfile(h5py_all_images_file):

    path_img_folder = wd + "/img_align_celeba/"
    
    images = np.zeros((N_all, NC, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    
    print("\n Begin loading image >>>")
    for i in tqdm(range(N_all)):
        filename_cur = image_filename[i]
        
        # load, resieze and store image to numpy array
        image = Image.open(path_img_folder + filename_cur)
        if Resized_Method == "BILINEAR":
            image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        elif Resized_Method == "LANCZOS": #best quality, worst performance
            image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        image = np.array(image).transpose(2,0,1) #C,H,W
        images[i] = image.astype(np.uint8)
    # end for i 
        
    with h5py.File(h5py_all_images_file, "w") as f:
        f.create_dataset('images', data = images, dtype='uint8')
else:
    with h5py.File(h5py_all_images_file, "r") as f:
        images = f['images'][:]
        
    
labels = np.zeros(N_all)    
print("\n Begin assigning labels >>>")
for i in tqdm(range(N_all)):
    filename_cur = image_filename[i]
    idx_sample = image_filename.index(filename_cur) #index of current image in attr_data
    labels[i] = attr_comb_to_label[tuple(attr_data[idx_sample][sel_indices])]


## compute number of samples in each class
for i in range(N_CLASS):
    num_samples_cur = sum(labels==i)
    print("Class %d, %s: %d \r" % (i, (all_combinations[i]), num_samples_cur))


######################
## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
### combine some class
#combine_indices = [[1, 2], [5, 6]]
#for i in range(len(combine_indices)):
#    min_label = min(combine_indices[i])
#    for j in range(len(combine_indices[i])):
#        indx_i = np.where(labels==combine_indices[i][j])[0]
#        labels[indx_i] = min_label
### relabeling
#unique_labels_old = list(set(labels))
#for i in range(len(unique_labels_old)):
#    labels[labels==unique_labels_old[i]] = i
#N_CLASS = len(set(labels))


#split into training and testing set
NValid = 10000
NTrain = N_all - NValid
indx_valid = np.arange(N_all)
np.random.shuffle(indx_valid)
indx_valid = indx_valid[0:NValid]
indx_all = set(np.arange(N_all))
indx_train = np.array(list(indx_all.difference(indx_valid)))
assert set(indx_train).union(set(indx_valid)) == indx_all

images_train = images[indx_train]
labels_train = labels[indx_train]
images_valid = images[indx_valid]
labels_valid = labels[indx_valid]
attr_train = attr_data[indx_train]
attr_valid = attr_data[indx_valid]


## compute number of samples in each class
for i in range(N_CLASS):
    train_num_samples_cur = sum(labels_train==i)
    print("Train, Class %d, %s: %d \r" % (i, (all_combinations[i]), train_num_samples_cur))
for i in range(N_CLASS):
    valid_num_samples_cur = sum(labels_valid==i)
    print("Valid, Class %d, %s: %d \r" % (i, (all_combinations[i]), valid_num_samples_cur))



###############################################################################
# 3. dump to h5py file
h5py_file = './celeba_64x64.h5'
with h5py.File(h5py_file, "w") as f:
#    f.create_dataset('images', data = images, dtype='uint8')
#    f.create_dataset('labels', data = labels, dtype='int')
    f.create_dataset('images_train', data = images_train, dtype='uint8')
    f.create_dataset('labels_train', data = labels_train, dtype='int')
    f.create_dataset('images_valid', data = images_valid, dtype='uint8')
    f.create_dataset('labels_valid', data = labels_valid, dtype='int')
    f.create_dataset('attr_train', data = attr_train, dtype='int')
    f.create_dataset('attr_valid', data = attr_valid, dtype='int')
    



#test h5 file
hf = h5py.File(h5py_file, 'r')
#img_v = hf['images_valid'][:]
#attr_t = hf['attr_train'][:]
img_v = hf['images_valid'][np.random.randint(0,len(images))]
img_v  = Image.fromarray(np.transpose(img_v, (1,2,0))); img_v .show()
hf.close()

