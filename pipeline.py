import os
import glob
import random
import torch
from torchvision import transforms as transforms
import numpy as np
from PIL import Image


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, opt, level=None):
        super(CustomDataset, self).__init__()
        dataset_name = opt.dataset_name
        is_train = opt.is_train
        format = opt.format

        dataset_dir = os.path.join('./datasets', dataset_name)

        assert os.path.isdir(dataset_dir), print(
            "{} does not exist. Please check your dataset directory.".format(dataset_dir))

        if dataset_name == 'Cityscapes':
            if is_train:
                self.label_path_list = sorted(
                    glob.glob(os.path.join(dataset_dir, 'Train', 'Input', 'LabelMap', '*.' + format)))
                self.instance_path_list = sorted(
                    glob.glob(os.path.join(dataset_dir, 'Train', 'Input', 'InstanceMap', '*.' + format)))
                self.target_path_list = sorted(glob.glob(os.path.join(dataset_dir, 'Train', 'Target', '*.' + format)))

            else:
                self.label_path_list = sorted(
                    glob.glob(os.path.join(dataset_dir, 'Test', 'Input', 'LabelMap', '*.' + format)))
                self.instance_path_list = sorted(
                    glob.glob(os.path.join(dataset_dir, 'Test', 'Input', 'InstanceMap', '*.' + format)))
                self.target_path_list = sorted(glob.glob(os.path.join(dataset_dir, 'Test', 'Target', '*.' + format)))

        elif dataset_name == 'Custom':
            if is_train:
                self.label_path_list = sorted(
                    glob.glob(os.path.join(dataset_dir, 'Train', 'Input', 'LabelMap', '*.' + format)))
                self.target_path_list = sorted(glob.glob(os.path.join(dataset_dir, 'Train', 'Target', '*.' + format)))

            else:
                self.label_path_list = sorted(
                    glob.glob(os.path.join(dataset_dir, 'Test', 'Input', 'LabelMap', '*.' + format)))
                self.target_path_list = sorted(glob.glob(os.path.join(dataset_dir, 'Test', 'Target', '*.' + format)))

        else:
            raise NotImplementedError("Please check dataset_name. It should be in ['Cityscapes', 'Custom'].")

        self.condition = opt.C_condition
        self.dataset_name = dataset_name
        self.flip = opt.flip
        self.image_size = opt.image_size
        self.is_train = is_train
        self.min_image_size = opt.min_image_size
        self.progression = opt.progression

        self.level = opt.n_downsample if level is None else level

    def get_transform(self, normalize=False, level_size=False):
        transform_list = []

        if not level_size:
            transform_list += [transforms.Resize(self.image_size, interpolation=Image.NEAREST)]

        elif level_size:
            transform_list += [transforms.Resize((2 ** int((np.log2(self.min_image_size[0]) + self.level)),
                                                  2 ** int(np.log2(self.min_image_size[1]) + self.level)),
                                                 interpolation=Image.NEAREST)]

        if self.is_train and self.coin:
            transform_list.append(transforms.Lambda(lambda x: self.__flip(x)))

        transform_list.append(transforms.ToTensor())

        if normalize:
            transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        return transforms.Compose(transform_list)

    @staticmethod
    def get_edges(instance_tensor):
        edge = torch.ByteTensor(instance_tensor.shape).zero_()
        edge[:, :, 1:] = edge[:, :, 1:] | (instance_tensor[:, :, 1:] != instance_tensor[:, :, :-1])
        edge[:, :, :-1] = edge[:, :, :-1] | (instance_tensor[:, :, 1:] != instance_tensor[:, :, :-1])
        edge[:, 1:, :] = edge[:, 1:, :] | (instance_tensor[:, 1:, :] != instance_tensor[:, :-1, :])
        edge[:, :-1, :] = edge[:, :-1, :] | (instance_tensor[:, 1:, :] != instance_tensor[:, :-1, :])

        return edge.float()

    @staticmethod
    def __flip(x):
        return x.transpose(Image.FLIP_LEFT_RIGHT)

    def encode_input(self, label_tensor, instance_tensor=None):
        if self.dataset_name == 'Cityscapes':
            max_label_index = 35
            shape = label_tensor.shape
            one_hot_shape = (max_label_index, shape[1], shape[2])
            label = torch.FloatTensor(torch.Size(one_hot_shape)).zero_()
            label = label.scatter_(dim=0, index=label_tensor.long(), src=torch.tensor(1.0))

            edge = self.get_edges(instance_tensor)

            input_tensor = torch.cat([label, edge], dim=0)

            return input_tensor

        elif self.dataset_name == 'Custom':
            return label_tensor

    def __getitem__(self, index):
        if self.dataset_name == 'Cityscapes':
            if self.flip and self.is_train:
                self.coin = random.random() > 0.5

            label_array = Image.open(self.label_path_list[index])
            label_tensor = self.get_transform()(label_array) * 255.0

            instance_array = Image.open(self.instance_path_list[index])
            instance_tensor = self.get_transform()(instance_array)

            input_tensor = self.encode_input(label_tensor, instance_tensor)

            target_array = Image.open(self.target_path_list[index])
            target_tensor = self.get_transform(normalize=True, level_size=True)(target_array)

            data_dict = {'input_tensor': input_tensor, 'target_tensor': target_tensor}

            if self.progression or self.condition:
                C_label_tensor = self.get_transform(level_size=True)(label_array) * 255.0
                C_instance_tensor = self.get_transform(level_size=True)(instance_array)
                C_input_tensor = self.encode_input(C_label_tensor, C_instance_tensor)

                data_dict.update({'C_input_tensor': C_input_tensor})

        elif self.dataset_name == 'Custom':
            label_array = Image.open(self.label_path_list[index])
            label_tensor = self.get_transform(normalize=True)(label_array)

            input_tensor = self.encode_input(label_tensor)

            target_array = Image.open(self.target_path_list[index])
            target_tensor = self.get_transform(normalize=True, level_size=True)(target_array)

            data_dict = {'input_tensor': input_tensor, 'target_tensor': target_tensor}

            if self.progression or self.condition:
                C_label_tensor = self.get_transform(normalize=True, level_size=True)(label_array)
                C_input_tensor = self.encode_input(C_label_tensor)

                data_dict.update({'C_input_tensor': C_input_tensor})

        else:
            raise NotImplementedError("Please check dataset_name. It should be in ['Cityscapes', 'Custom'].")

        return data_dict

    def __len__(self):
        return len(self.label_path_list)
