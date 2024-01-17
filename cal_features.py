#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/17 9:17
# @Author  : Xiao Li
# @File    : cal_features.py
import os
import cv2
import argparse
import skimage
import math
import numpy as np
import pandas as pd
from glob import glob
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import label2rgb
from skimage.measure import label, regionprops, shannon_entropy
from skimage.feature import local_binary_pattern, peak_local_max, graycomatrix, graycoprops
from skimage.filters.rank import entropy
from skimage.filters import gabor
from skimage.segmentation import watershed, random_walker
from skimage.morphology import disk, erosion, remove_small_objects
import skimage.morphology as morphology
#from dynamic_watershed import *

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', type=str, required=True)
    parse.add_argument('--out_path', type=str, required=True)
    args = parse.parse_args()
    return args

def getData(args):
    root = args.data_path
    img_paths = glob(root + '/images/*')
    mask_paths = glob(root + '/masks/*')
    #img_paths = glob(root + '/images/*')
    #mask_paths = glob(root + '/masks/*')
    #print(img_paths, mask_paths)
    assert len(img_paths) == len(mask_paths), '图片数和掩码数不符'
    return img_paths, mask_paths

def compute_entropy(image):
    im_aux = image / np.max(image)
    filtered = entropy(im_aux, disk(3))
    return filtered

def compute_energy(image):
    aux = np.sqrt((np.real(image)**2) + (np.imag(image)**2))
    return np.sum(aux**2)

def compute_amplitude(image):
    aux = np.sqrt((np.real(image)**2) + (np.imag(image)**2))
    return np.mean(aux)

def Watershed_Marker(mask):
    distance = ndi.distance_transform_edt(mask)
    # local_maxi = peak_local_max(distance, labels=mask, footprint=np.ones((3, 3)), indices=False,exclude_border=True)
    local_maxi = peak_local_max(distance, labels=mask, min_distance=5)
    # markers = ndi.label(local_maxi)[0]
    # markers = ndi.label(local_maxi, structure=np.ones((3,3)))[0]
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_maxi.T)] = True
    markers = label(local_max_mask, connectivity=1)
    # mask = watershed(-distance, markers, mask=mask, connectivity=2)
    labels = watershed(-distance, markers, mask=mask)
    labels = remove_small_objects(labels, 10)
    labelsrgb = label2rgb(labels)
    return labels, labelsrgb


def condition_erosion(mask, erosion_structure, threshold):
    mask_process = np.zeros(np.shape(mask))
    mask_label, N = label(mask, return_num=True)
    for i in range(1, N + 1):
        mask_temp = (mask_label == i)
        while np.sum(mask_temp) >= threshold:
            mask_temp = erosion(mask_temp, erosion_structure)
        mask_process = mask_process + mask_temp
    return mask_process

def Watershed_Condition_erosion(mask):
    fine_structure = morphology.diamond(1)
    #fine_structure[2,0] = 0
    #fine_structure[2,4] = 0
    #fine_structure = np.array([[1,1,1],[1,1,1],[1,1,1]])
    coarse_structure = morphology.diamond(3)
    coarse_structure[0,3] = 0
    coarse_structure[6,3] = 0
    #coarse_structure[3, 0] = 0
    #coarse_structure[3, 6] = 0

    #coarse_structure1 = morphology.diamond(3)
    #coarse_structure1[0, 3] = 0
    #coarse_structure1[6, 3] = 0
    #print(fine_structure)
    #print(coarse_structure)

    # ==========step1 coarse erosion============= 200
    seed_mask = condition_erosion(mask, coarse_structure, 200)
    #plt.imshow(seed_mask)
    #plt.show()
    '''
    fig = plt.figure(figsize=(8, 8))  # 4,4
    plt.axis('off')
    plt.imshow(seed_mask)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.savefig('cal_features/watershed1106/coarse.png', dpi=64)  # 64
    '''

    #seed_mask1 = condition_erosion(mask, coarse_structure, 200)
    #plt.imshow(seed_mask1)
    #plt.show()
    # ==========step2 fine erosion============= 80
    seed_mask = condition_erosion(seed_mask, fine_structure, 80)
    #plt.imshow(seed_mask)
    #plt.show()
    '''
    fig = plt.figure(figsize=(8, 8))  # 4,4
    plt.axis('off')
    plt.imshow(seed_mask)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.savefig('cal_features/watershed1106/fine.png', dpi=64)  # 64
    '''

    #seed_mask1 = condition_erosion(seed_mask, fine_structure1, 50)
    #plt.imshow(seed_mask1)
    #plt.show()

    distance = ndi.distance_transform_edt(mask)
    markers = label(seed_mask, connectivity=1)
    #plt.imshow(markers)
    #plt.show()
    labels = watershed(-distance, markers, mask=mask)
    labelsrgb = label2rgb(labels)
    #plt.imshow(labelsrgb)
    #plt.show()
    #    markersrgb = label2rgb(markers,bg_label = 0, bg_color=(0.2, 0.5, 0.6))
    return labels, labelsrgb

'''
def Rw_watershed(mask):
    fine_structure = morphology.diamond(1)
    coarse_structure = morphology.diamond(3)
    coarse_structure[3, 0] = 0
    coarse_structure[3, 6] = 0
    # coarse_structure[0, 3] = 0
    # coarse_structure[0, 6] = 0
    # print(fine_structure)
    # print(coarse_structure)

    # ==========step1 coarse erosion=============
    seed_mask = condition_erosion(mask, coarse_structure, 200)
    plt.imshow(seed_mask)
    plt.show()
    # ==========step2 fine erosion=============
    seed_mask = condition_erosion(seed_mask, fine_structure, 40)
    plt.imshow(seed_mask)
    plt.show()

    kernel = np.ones((3, 3), np.uint8)
    bg = cv2.dilate(mask, kernel, iterations=3)
    seed_mask[bg == 0] = 2

    labels = random_walker(mask, seed_mask, beta=130, mode='bf')
    #labels[labels == 2] = 0
    labelsrgb = label2rgb(labels)
    return labels, labelsrgb
'''

def cal_features(img_paths, mask_paths):
    features = pd.DataFrame(columns=['Image', 'Label','Centroid_x', 'Centroid_y', 'Centroid_weighted_x',
                                     'Centroid_weighted_y', 'Area', 'Area_bbox', 'Area_convex', 'Area_filled',
                                     'Axis_major_length', 'Axis_minor_length',  'Eccentricity', 'Equivalent_diameter_area', 'Euler_number',
                                     'Extent', 'Feret_diameter_max', 'Inertia_tensor-0-0', 'Inertia_tensor-0-1','Inertia_tensor-1-0',
                                     'Inertia_tensor-1-1', 'Inertia_tensor_eigvals-0', 'Inertia_tensor_eigvals-1', 'Intensity_max',
                                     'Intensity_mean', 'Intensity_min', 'Moments-0-0', 'Moments-0-1',
                                     'Moments-0-2', 'Moments-0-3','Moments-1-0', 'Moments-1-1', 'Moments-1-2', 'Moments-1-3',
                                     'Moments-2-0', 'Moments-2-1','Moments-2-2', 'Moments-2-3', 'Moments-3-0', 'Moments-3-1',
                                     'Moments-3-2', 'Moments-3-3',
                                     'Moments_central-0-0', 'Moments_central-0-1', 'Moments_central-0-2', 'Moments_central-0-3',
                                     'Moments_central-1-0', 'Moments_central-1-1', 'Moments_central-1-2', 'Moments_central-1-3',
                                     'Moments_central-2-0', 'Moments_central-2-1', 'Moments_central-2-2', 'Moments_central-2-3',
                                     'Moments_central-3-0', 'Moments_central-3-1', 'Moments_central-3-2', 'Moments_central-3-3',
                                     'Moments_hu-0', 'Moments_hu-1', 'Moments_hu-2', 'Moments_hu-3', 'Moments_hu-4', 'Moments_hu-5',
                                     'Moments_hu-6', 'Orientation', 'Perimeter', 'Perimeter_crofton', 'Solidity', 'Form_factor',
                                     'SER-Spot', 'SER-Edge', 'SER-Ridge', 'SER-Valley', 'SER-Saddle',
                                     'Local Binary Pattern', 'Entropy', 'Shannon Entropy', 'Gabor', 'Contrast-0-0','Contrast-0-1',
                                     'Contrast-0-2', 'Contrast-0-3', 'Dissimilarity-0-0', 'Dissimilarity-0-1', 'Dissimilarity-0-2',
                                     'Dissimilarity-0-3', 'Homogeneity-0-0', 'Homogeneity-0-1', 'Homogeneity-0-2', 'Homogeneity-0-3',
                                     'Energy-0-0', 'Energy-0-1','Energy-0-2', 'Energy-0-3', 'Correlation-0-0', 'Correlation-0-1',
                                     'Correlation-0-2', 'Correlation-0-3', 'ASM-0-0', 'ASM-0-1','ASM-0-2', 'ASM-0-3'])

    for img_path, mask_path in tqdm(zip(img_paths, mask_paths)):
        #print(img_path)
        #print(img_path.split(os.sep))
        pic = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        #pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #plt.imshow(mask)
        #plt.show()
        labels, labelsrgb = Watershed_Condition_erosion(mask)
        #labels, labelsrgb = Rw_watershed(mask)
        #labels, labelsrgb = Watershed_Marker(mask)
        #labels = post_process(mask, 1, thresh=0.99)
        #labelsrgb = label2rgb(labels)

        '''
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title('input')
        plt.imshow(pic, plt.cm.gray)
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title('entropy')
        plt.imshow(entropy(pic, disk(3)), cmap='Greys_r')
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title('lbp')
        plt.imshow(local_binary_pattern(pic, 8, 1, 'uniform'), cmap='Greys_r')
        plt.show()
        '''

        '''
        mask_label = label(mask)
        mask_label = skimage.morphology.remove_small_objects(mask_label, min_size = 50)
        mask_label = clear_border(mask_label)
        print(mask_label.shape, pic.shape)
        plt.imshow(mask_label)
        plt.show()
        '''

        mask_label = skimage.morphology.remove_small_objects(labels, min_size=80)
        mask_label = clear_border(mask_label)

        save_path = os.path.join(args.out_path, str('watershed1106'), mask_path.split(os.sep)[-1])
        fig = plt.figure(figsize=(8,8))   #4,4
        plt.axis('off')
        plt.imshow(label2rgb(mask_label))
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        fig.savefig(save_path, dpi=64) #64
        #plt.show()


        props= regionprops(mask_label, intensity_image = pic)
        #print(props)
        cnt = 0
        for prop in props:
            cnt += 1
            #print('inertia_tensor', prop.inertia_tensor)
            #print('inertia_tensor_eigvals', prop.inertia_tensor_eigvals)
            #print('moments', prop.moments)
            #print('moments_central', prop.moments_central)
            #print('moments_hu', prop.moments_hu)

            glcm = graycomatrix(prop.intensity_image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
            gradient_x = ndi.gaussian_filter(prop.intensity_image, sigma=1.0, order=(0,1))
            gradient_y = ndi.gaussian_filter(prop.intensity_image, sigma=1.0, order=(1,0))

            res = {#'Num_pixels':prop.num_pixels,
                   'Image':img_path.split(os.sep)[-1],
                   'Label': cnt,
                   'Centroid_x': prop.centroid[1],
                   'Centroid_y': prop.centroid[0],
                   'Centroid_weighted_x': prop.centroid_weighted[1],
                   'Centroid_weighted_y': prop.centroid_weighted[0],
                   'Area':prop.area,
                   'Area_bbox':prop.area_bbox,
                   'Area_convex':prop.area_convex,
                   'Area_filled':prop.area_filled,
                   'Axis_major_length':prop.axis_major_length,
                   'Axis_minor_length':prop.axis_minor_length,
                   'Eccentricity':prop.eccentricity,
                   'Equivalent_diameter_area':prop.equivalent_diameter_area,
                   'Euler_number':prop.euler_number,
                   'Extent':prop.extent,
                   'Feret_diameter_max':prop.feret_diameter_max,
                   'Inertia_tensor-0-0':prop.inertia_tensor[0][0],
                   'Inertia_tensor-0-1':prop.inertia_tensor[0][1],
                   'Inertia_tensor-1-0':prop.inertia_tensor[1][0],
                   'Inertia_tensor-1-1':prop.inertia_tensor[1][1],
                   'Inertia_tensor_eigvals-0':prop.inertia_tensor_eigvals[0],
                   'Inertia_tensor_eigvals-1':prop.inertia_tensor_eigvals[1],
                   'Intensity_max':prop.intensity_max,
                   'Intensity_mean':prop.intensity_mean,
                   'Intensity_min':prop.intensity_min,
                   'Moments-0-0':prop.moments[0][0],
                   'Moments-0-1':prop.moments[0][1],
                   'Moments-0-2':prop.moments[0][2],
                   'Moments-0-3':prop.moments[0][3],
                   'Moments-1-0':prop.moments[1][0],
                   'Moments-1-1':prop.moments[1][1],
                   'Moments-1-2':prop.moments[1][2],
                   'Moments-1-3':prop.moments[1][3],
                   'Moments-2-0':prop.moments[2][0],
                   'Moments-2-1':prop.moments[2][1],
                   'Moments-2-2':prop.moments[2][2],
                   'Moments-2-3':prop.moments[2][3],
                   'Moments-3-0':prop.moments[3][0],
                   'Moments-3-1':prop.moments[3][1],
                   'Moments-3-2':prop.moments[3][2],
                   'Moments-3-3':prop.moments[3][3],
                   'Moments_central-0-0':prop.moments_central[0][0],
                   'Moments_central-0-1':prop.moments_central[0][1],
                   'Moments_central-0-2':prop.moments_central[0][2],
                   'Moments_central-0-3':prop.moments_central[0][3],
                   'Moments_central-1-0':prop.moments_central[1][0],
                   'Moments_central-1-1':prop.moments_central[1][1],
                   'Moments_central-1-2':prop.moments_central[1][2],
                   'Moments_central-1-3':prop.moments_central[1][3],
                   'Moments_central-2-0':prop.moments_central[2][0],
                   'Moments_central-2-1':prop.moments_central[2][1],
                   'Moments_central-2-2':prop.moments_central[2][2],
                   'Moments_central-2-3':prop.moments_central[2][3],
                   'Moments_central-3-0':prop.moments_central[3][0],
                   'Moments_central-3-1':prop.moments_central[3][1],
                   'Moments_central-3-2':prop.moments_central[3][2],
                   'Moments_central-3-3':prop.moments_central[3][3],
                   'Moments_hu-0':prop.moments_hu[0],
                   'Moments_hu-1':prop.moments_hu[1],
                   'Moments_hu-2':prop.moments_hu[2],
                   'Moments_hu-3':prop.moments_hu[3],
                   'Moments_hu-4':prop.moments_hu[4],
                   'Moments_hu-5':prop.moments_hu[5],
                   'Moments_hu-6':prop.moments_hu[6],
                   'Orientation':prop.orientation,
                   'Perimeter':prop.perimeter,
                   'Perimeter_crofton':prop.perimeter_crofton,
                   'Solidity':prop.solidity,
                   'Form_factor':4*math.pi*prop.area/(prop.perimeter**2),
                   'SER-Spot': np.mean((np.abs(gradient_x) + np.abs(gradient_y)) / 2),
                   'SER-Edge': np.mean(np.sqrt(gradient_x**2 + gradient_y**2)),
                   'SER-Ridge': np.mean(np.maximum(gradient_x, gradient_y)),
                   'SER-Valley': np.mean(np.minimum(gradient_x, gradient_y)),
                   'SER-Saddle': np.mean(np.abs(gradient_x - gradient_y)),
                   'Local Binary Pattern':np.mean(local_binary_pattern(prop.intensity_image, 8, 1, 'uniform')),
                   'Entropy':np.mean(compute_entropy(prop.intensity_image)),
                   'Shannon Entropy':shannon_entropy(prop.intensity_image),
                   'Gabor':np.mean(gabor(prop.intensity_image, frequency=0.6)),
                   'Contrast-0-0': graycoprops(glcm, 'contrast')[0][0],
                   'Contrast-0-1': graycoprops(glcm, 'contrast')[0][1],
                   'Contrast-0-2': graycoprops(glcm, 'contrast')[0][2],
                   'Contrast-0-3': graycoprops(glcm, 'contrast')[0][3],
                   'Dissimilarity-0-0': graycoprops(glcm, 'dissimilarity')[0][0],
                   'Dissimilarity-0-1': graycoprops(glcm, 'dissimilarity')[0][1],
                   'Dissimilarity-0-2': graycoprops(glcm, 'dissimilarity')[0][2],
                   'Dissimilarity-0-3': graycoprops(glcm, 'dissimilarity')[0][3],
                   'Homogeneity-0-0': graycoprops(glcm, 'homogeneity')[0][0],
                   'Homogeneity-0-1': graycoprops(glcm, 'homogeneity')[0][1],
                   'Homogeneity-0-2': graycoprops(glcm, 'homogeneity')[0][2],
                   'Homogeneity-0-3': graycoprops(glcm, 'homogeneity')[0][3],
                   'Energy-0-0': graycoprops(glcm, 'energy')[0][0],
                   'Energy-0-1': graycoprops(glcm, 'energy')[0][1],
                   'Energy-0-2': graycoprops(glcm, 'energy')[0][2],
                   'Energy-0-3': graycoprops(glcm, 'energy')[0][3],
                   'Correlation-0-0': graycoprops(glcm, 'correlation')[0][0],
                   'Correlation-0-1': graycoprops(glcm, 'correlation')[0][1],
                   'Correlation-0-2': graycoprops(glcm, 'correlation')[0][2],
                   'Correlation-0-3': graycoprops(glcm, 'correlation')[0][3],
                   'ASM-0-0': graycoprops(glcm, 'ASM')[0][0],
                   'ASM-0-1': graycoprops(glcm, 'ASM')[0][1],
                   'ASM-0-2': graycoprops(glcm, 'ASM')[0][2],
                   'ASM-0-3': graycoprops(glcm, 'ASM')[0][3],
            }
            #print(res)
            features = pd.concat([features, pd.DataFrame([res])], ignore_index=True)
            #features = features.append(res, ignore_index=True)

        #dir = os.path.join(args.out_path, 'features-0914-ch2.csv')
        #features.to_csv(dir, header=True, index=True)
    

    return features



if __name__ == '__main__':
    args = getArgs()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, str('watershed1106'))):
        os.makedirs(os.path.join(args.out_path, str('watershed1106')))
        print(os.path.join(args.out_path, str('watershed1106')))

    img_paths, mask_paths = getData(args)
    features_df = cal_features(img_paths, mask_paths)
    print(features_df)

    dir = os.path.join(args.out_path, 'features-1106.csv')
    features_df.to_csv(dir, header=True, index=True)