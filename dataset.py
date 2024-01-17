# -*- coding: utf-8 -*-
# @Time    : 2023/2/20 10:57
# @Author  : Xiao Li
# @File    : dataset.py
import os
import cv2
import numpy as np
import random
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import PIL.Image as Image
from skimage.io import imread
import imageio

class dsb2018CellDataset(data.Dataset):
    def __init__(self, img_paths, mask_paths, augmentation=None, preprocessing=None, x_transform=None, target_transform=None):
        #self.state = state
        #self.root = r'D:\HCS\pre\dsb2018Cell'
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        #self.train_img_paths, self.val_img_paths = None, None
        #self.train_mask_paths, self.val_mask_paths = None, None
        #self.pics, self.masks
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.x_transform = x_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        pic_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
        #pic = pic.astype('float32') / 255
        #mask = mask.astype('float32') / 255

        if self.augmentation:
            augments = self.augmentation(image=pic, mask=mask)
            pic, mask = augments['image'],augments['mask']
        if self.preprocessing:
            #print(pic.size, mask.size)
            sample = self.preprocessing(image=pic, mask=mask)
            pic, mask = sample['image'], sample['mask']
            #print(pic.size, mask.size)
        if self.x_transform is not None:
            pic = self.x_transform(pic)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return pic, mask, pic_path, mask_path

    def __len__(self):
        return len(self.img_paths)

class PanNukeDataset(data.Dataset):
    def __init__(self, img_paths, mask_paths, augmentation=None, preprocessing=None, x_transform=None, target_transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.x_transform = x_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        pic_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)

        if self.augmentation:
            augments = self.augmentation(image=pic, mask=mask)
            pic, mask = augments['image'],augments['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=pic, mask=mask)
            pic, mask = sample['image'], sample['mask']
        if self.x_transform is not None:
            pic = self.x_transform(pic)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return pic, mask, pic_path, mask_path

    def __len__(self):
        return len(self.img_paths)

class myDataset(data.Dataset):
    def __init__(self, img_paths, mask_paths, augmentation=None, preprocessing=None, x_transform=None, target_transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.x_transform = x_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        pic_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
        #pic = pic.astype('float32') / 255
        #mask = mask.astype('float32') / 255
        #print(pic.shape, mask.shape)

        if self.augmentation:
            augments = self.augmentation(image=pic, mask=mask)
            pic, mask = augments['image'],augments['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=pic, mask=mask)
            pic, mask = sample['image'], sample['mask']
        if self.x_transform is not None:
            pic = self.x_transform(pic)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return pic, mask, pic_path, mask_path

    def __len__(self):
        return len(self.img_paths)