from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
from utils import *

class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            images_list = sorted(os.listdir(path_Data+'val/images/'))
            masks_list = sorted(os.listdir(path_Data+'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)

class NPY_datasets2(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets2, self)
        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer2
        else:
            images_list = sorted(os.listdir(path_Data+'val/images/'))
            masks_list = sorted(os.listdir(path_Data+'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
    
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import os
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset

class NPY_datasets3(Dataset):
    def __init__(self, path_Data, config, train=True, apply_mixup=True):
        """
        Dataset for segmentation tasks with optional MixUp augmentation.

        Parameters:
        - path_Data: Path to the dataset.
        - config: Configuration containing transformations.
        - train: If True, uses training set; otherwise, uses validation set.
        - apply_mixup: Whether to apply MixUp augmentation.
        """
        super(NPY_datasets3, self).__init__()
        self.train = train
        self.apply_mixup = apply_mixup

        dataset_type = 'train' if train else 'val'
        images_list = sorted(os.listdir(os.path.join(path_Data, f'{dataset_type}/images/')))
        masks_list = sorted(os.listdir(os.path.join(path_Data, f'{dataset_type}/masks/')))

        self.transformer = config.train_transformer if train else config.test_transformer

        # Load image-mask pairs
        self.data = [
            (
                os.path.join(path_Data, f'{dataset_type}/images/', img),
                os.path.join(path_Data, f'{dataset_type}/masks/', msk)
            )
            for img, msk in zip(images_list, masks_list)
        ]

        # Initialize MixUp augmentation if enabled and dataset is for training
        self.mixup = MixUpSegmentation(self) if self.apply_mixup and train else None

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        
        # Load image and mask as numpy arrays
        img = np.array(Image.open(img_path).convert('RGB'))  # Convert image to RGB
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255  # Convert mask to single channel

        # Ensure transformer receives the data as tuples of (image, mask) arrays
        sample = (img, msk)
        
        # Apply transformations to the image and mask
        img, msk = self.transformer(sample)  # Apply transformations

        # Apply MixUp augmentation with probability p if enabled
        if self.mixup:
            img, msk = self.mixup((img, msk))

        return img, msk

    def __len__(self):
        return len(self.data)




class NPY_datasets4(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets4, self)
        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer4
        else:
            images_list = sorted(os.listdir(path_Data+'val/images/'))
            masks_list = sorted(os.listdir(path_Data+'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
    
    
    
class NPY_datasets5(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets5, self)
        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer5
        else:
            images_list = sorted(os.listdir(path_Data+'val/images/'))
            masks_list = sorted(os.listdir(path_Data+'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
    
       
    

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
        

import os
from torchvision import transforms


class NPY_datasets3(Dataset):
    def __init__(self, path_Data, config, train=True, apply_mixup=True):
        """
        Dataset for segmentation tasks with optional MixUp augmentation.

        Parameters:
        - path_Data: Path to the dataset.
        - config: Configuration containing transformations.
        - train: If True, uses training set; otherwise, uses validation set.
        - apply_mixup: Whether to apply MixUp augmentation.
        """
        super(NPY_datasets3, self).__init__()
        self.train = train
        self.apply_mixup = apply_mixup

        dataset_type = 'train' if train else 'val'
        images_list = sorted(os.listdir(os.path.join(path_Data, f'{dataset_type}/images/')))
        masks_list = sorted(os.listdir(os.path.join(path_Data, f'{dataset_type}/masks/')))

        self.transformer = config.train_transformer if train else config.test_transformer

        # Load image-mask pairs
        self.data = [
            (
                os.path.join(path_Data, f'{dataset_type}/images/', img),
                os.path.join(path_Data, f'{dataset_type}/masks/', msk)
            )
            for img, msk in zip(images_list, masks_list)
        ]

        # Initialize MixUp augmentation if enabled and dataset is for training
        self.mixup = MixUpSegmentation(self) if self.apply_mixup and train else None

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        
        # Load image and mask as numpy arrays
        img = np.array(Image.open(img_path).convert('RGB'))  # Convert image to RGB
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255  # Convert mask to single channel

        # Ensure transformer receives the data as tuples of (image, mask) arrays
        sample = (img, msk)
        
        # Apply transformations to the image and mask
        img, msk = self.transformer(sample)  # Apply transformations

        # Apply MixUp augmentation only to the image, not the mask
        if self.mixup:
            img, _ = self.mixup((img, msk))  # Only mix the image, not the mask
        
        # Ensure mask values are within the valid range [0, 1] for binary masks
        msk = np.clip(msk, 0, 1)  # Clip mask to binary range [0, 1] (no gray regions)

        return img, msk

    def __len__(self):
        return len(self.data)
