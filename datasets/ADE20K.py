import os
import h5py
from scipy import io
from glob import glob
import tables
import h5py


dataset_dir = '/Users/noel/Projects/DeepLearning/Dataset/ADE20K/ADE20K_2016_07_26'
mat = os.path.join(dataset_dir, 'index_ade20k.mat')
training_set_dir = os.path.join(dataset_dir, 'training')
val_set_dir = os.path.join(dataset_dir, 'validation')

# mat_contents = io.loadmat(mat)

with h5py.File(mat) as f:
    print(f.keys())


# file = tables.open_file(mat)
# print(file)
