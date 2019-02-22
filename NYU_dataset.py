import os
import png
from glob import glob
import torch
from torchvision.transforms import ToPILImage
import numpy as np
import h5py
from PIL import Image

dataset_dir = '/Users/noel/Projects/DeepLearning/Dataset/NYU'
inst_dir = os.path.join(dataset_dir, 'instance')
label_dir = os.path.join(dataset_dir, 'label')
mat_file = os.path.join(dataset_dir, 'nyu_depth_v2_labeled.mat')
toolbox = os.path.join(dataset_dir, 'toolbox_nyu_depth_v2', 'get_instance_masks.m')

dst_inst_train = os.path.join(dataset_dir, 'Train', 'Input', 'InstanceMap')
dst_label_train = os.path.join(dataset_dir, 'Train', 'Input', 'LabelMap')
dst_inst_test = os.path.join(dataset_dir, 'Test', 'Input', 'InstanceMap')
dst_label_test = os.path.join(dataset_dir, 'Test', 'Input', 'LabelMap')

with h5py.File(mat_file, 'r') as f:
    print(f.keys())

    for i in range(len(f['labels'])):
        label = np.array((f['labels'][i]))
        inst = Image.fromarray(np.array(f['instances'][i]).transpose())
        inst.save(os.path.join(dst_inst_train if i < 1200 else dst_inst_test, '{}.png'.format(i + 1)))

        with open(os.path.join(dst_label_train if i < 1200 else dst_label_test, '{}.png'.format(i + 1)), 'wb') as image_file:
            writer = png.Writer(width=label.shape[0], height=label.shape[1], bitdepth=16, greyscale=True)
            writer.write(image_file, np.array(label).transpose().tolist())


