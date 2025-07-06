import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import copy
import random

class FundusSeg_Loader(Dataset):
    def __init__(self, data_path, is_train, dataset_name, data_mean, data_std):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.data_mean = data_mean
        self.data_std = data_std

        if self.dataset_name == "drive" or self.dataset_name == "chase" or self.dataset_name == "rc-slo":
            self.imgs_path = glob.glob(os.path.join(data_path, 'img/*.tif'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))
        if self.dataset_name == "stare":
            self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.tif'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))
        if self.dataset_name == "rimone" or self.dataset_name == "refuge" or self.dataset_name == "refuge2":
            self.imgs_path = glob.glob(os.path.join(data_path, 'img/*.jpg'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))

        self.is_train = is_train

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        if self.dataset_name == "drive":
            label_path = image_path.replace('img', 'label')
            if self.is_train == 1:
                label_path = label_path.replace('_training.tif', '_manual1.tif') 
            else:
                label_path = label_path.replace('_test.tif', '_manual1.tif') 

        if self.dataset_name == "chase":
            label_path = image_path.replace('img', 'label')
            label_path = label_path.replace('.tif', '_1stHO.tif') 

        if self.dataset_name == "stare":
            label_path = image_path.replace('image', 'label')

        if self.dataset_name == "rc-slo":
            label_path = image_path.replace('img', 'label')

        if self.dataset_name == "rimone" or self.dataset_name == "refuge" or self.dataset_name == "refuge2":
            label_path = image_path.replace('img', 'label')
            label_path = label_path.replace('.jpg', '.tif') 

        image = Image.open(image_path)
        label = Image.open(label_path)
        label = label.convert('L')
        raw_height = image.size[1]
        raw_width = image.size[0]

        if self.dataset_name == "drive":
            image, label = self.padding_image(image, label, 594, 594)
        if self.dataset_name == "stare":
            image, label = self.padding_image(image, label, 702, 702)
        if self.dataset_name == "rimone" or self.dataset_name == "refuge" or self.dataset_name == "refuge2":
            image = image.resize((256,256))
            label = label.resize((256,256))

        # Online augmentation
        if self.is_train == 1:
            if torch.rand(1).item() <= 0.5:
                image, label = self.randomRotation(image, label)

            if torch.rand(1).item() <= 0.25:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)

            if torch.rand(1).item() <= 0.25:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                label = label.transpose(Image.FLIP_TOP_BOTTOM)

        image = np.asarray(image)
        label = np.asarray(label)

        if self.is_train == 1:
            image  = self.SR_LAB(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        label = label.reshape(1, label.shape[0], label.shape[1])
        label = np.array(label)

        if (label.max() == 255):
            label = label / 255

        # Normalize
        image = image / 255
        image = image.transpose(2, 0, 1)

        sp = image_path.split('/')
        filename = sp[len(sp)-1]
        filename = filename[0:len(filename)-4] # del .tif

        return image, label, filename, raw_height, raw_width

    def __len__(self):
        return len(self.imgs_path)

    def randomRotation(self, image, label, mode=Image.BICUBIC):
        random_angle = torch.randint(low=0,high=360,size=(1,1)).long().item()
        return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)

    def padding_image(self,image, label, pad_to_h, pad_to_w):
        new_image = Image.new('RGB', (pad_to_w, pad_to_h), (0, 0, 0))
        new_label = Image.new('P', (pad_to_w, pad_to_h), (0, 0, 0))
        new_image.paste(image, (0, 0))
        new_label.paste(label, (0, 0))
        return new_image, new_label

    # style randomization on the LAB color space
    def SR_LAB(self, image):
        LAB_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        LAB_image[:, :, 0] = self.random_transform_channel(LAB_image[:, :, 0])
        LAB_image[:, :, 1] = self.random_transform_channel(LAB_image[:, :, 1])
        LAB_image[:, :, 2] = self.random_transform_channel(LAB_image[:, :, 2])
        return LAB_image

    def random_transform_channel(self, channel):
        mean = 1  # 均值
        std_dev = 1.0  # 标准差
        a = np.clip(np.random.normal(mean, std_dev, 1), 0, 3)
        channel = channel.astype(np.float32) * a
        return channel
