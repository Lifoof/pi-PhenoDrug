# -*- coding: utf-8 -*-
# @Time    : 2023/2/22 9:44
# @Author  : Xiao Li
# @File    : utils.py
import math
import cv2
import os
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from skimage.io import imread
import imageio

'''
class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc
'''

def get_iou(mask_name,predict):
    image_mask = cv2.imread(mask_name,0)
    #print(image_mask)
    '''
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask,(576,576))
    '''
    #image_mask = mask
    # print(image.shape)
    height = predict.shape[0]
    weight = predict.shape[1]
    # print(height*weight)
    for row in range(height):
            for col in range(weight):
                if predict[row, col] < 0.5:  
                    predict[row, col] = 0
                else:
                    predict[row, col] = 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
            for col in range(weight_mask):
                if image_mask[row, col] < 125:   #由于mask图是黑白的灰度图，所以少于125的可以看作是黑色
                    image_mask[row, col] = 0
                else:
                    image_mask[row, col] = 1
    predict = predict.astype(np.int16)

    interArea = np.multiply(predict, image_mask)
    tem = predict + image_mask
    unionArea = tem - interArea
    inter = np.sum(interArea)
    union = np.sum(unionArea)
    #print(inter, union)
    iou_tem = inter / union

    return iou_tem

def get_dice(mask_name,predict):
    image_mask = cv2.imread(mask_name, 0)
    '''
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask,(576,576))
    '''
    height = predict.shape[0]
    weight = predict.shape[1]
    #o = 0
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:  
                predict[row, col] = 0
            else:
                predict[row, col] = 1
            #if predict[row, col] == 0 or predict[row, col] == 1:
                #o += 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
        for col in range(weight_mask):
            if image_mask[row, col] < 125:  
                image_mask[row, col] = 0
            else:
                image_mask[row, col] = 1
            #if image_mask[row, col] == 0 or image_mask[row, col] == 1:
                #o += 1
    predict = predict.astype(np.int16)
    intersection = (predict*image_mask).sum()
    dice = (2. *intersection) /(predict.sum()+image_mask.sum())
    return dice

def get_hd(mask_name,predict):
    image_mask = cv2.imread(mask_name, 0)
    # print(mask_name)
    # print(image_mask)
    '''
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask,(576,576))
    '''
    #image_mask = mask
    height = predict.shape[0]
    weight = predict.shape[1]
    #o = 0
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:  
                predict[row, col] = 0
            else:
                predict[row, col] = 1
            #if predict[row, col] == 0 or predict[row, col] == 1:
                #o += 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
        for col in range(weight_mask):
            if image_mask[row, col] < 125:  
                image_mask[row, col] = 0
            else:
                image_mask[row, col] = 1
            #if image_mask[row, col] == 0 or image_mask[row, col] == 1:
                #o += 1
    hd1 = directed_hausdorff(image_mask, predict)[0]
    hd2 = directed_hausdorff(predict, image_mask)[0]
    res = None
    if hd1>hd2 or hd1 == hd2:
        res=hd1
        return res
    else:
        res=hd2
        return res

def get_F1(mask_name,predict):
    TN, FN, TP, FP = 0, 0, 0, 0
    image_mask = cv2.imread(mask_name, 0)
    height = predict.shape[0]
    weight = predict.shape[1]
    # o = 0
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:
                predict[row, col] = 0
            else:
                predict[row, col] = 1
            # if predict[row, col] == 0 or predict[row, col] == 1:
            # o += 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
        for col in range(weight_mask):
            if image_mask[row, col] < 125:
                image_mask[row, col] = 0
            else:
                image_mask[row, col] = 1

    for row in range(height):
        for col in range(weight):
            if predict[row, col] == 1:
                if predict[row, col] == image_mask[row, col]:
                    TP = TP + 1
                else:
                    FP = FP + 1
            else:
                if predict[row, col] == image_mask[row, col]:
                    TN = TN + 1
                else:
                    FN = FN + 1

    Se = TP/(TP + FN)
    Sp = TN/(TN + FP)
    Pr = TP/(TP + FP)
    F1 = (2*Pr*Se)/(Pr+Se)
    G = math.sqrt(Se*Sp)
    IoU = (Pr*Se)/(Pr+Se-Pr*Se)
    DSC = (2*TP)/(2*TP + FP + FN)

    return F1

def show(predict):
    height = predict.shape[0]
    weight = predict.shape[1]
    for row in range(height):
        for col in range(weight):
            predict[row, col] *= 255
    plt.imshow(predict)
    plt.show()

def loss_plot(args,loss):
    num = args.epoch
    x = [i for i in range(num)]
    plot_save_path = r'result/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_loss = plot_save_path+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'_loss.jpg'
    plt.figure()
    plt.plot(x,loss,label='loss')
    plt.legend()
    plt.savefig(save_loss)

def metrics_plot(arg,name,*args):
    num = arg.epoch
    names = name.split('&')
    metrics_value = args
    i=0
    x = [i for i in range(num)]
    plot_save_path = r'result/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_metrics = plot_save_path + str(arg.arch) + '_' + str(arg.batch_size) + '_' + str(arg.dataset) + '_' + str(arg.epoch) + '_'+name+'.jpg'
    plt.figure()
    for l in metrics_value:
        plt.plot(x,l,label=str(names[i]))
        #plt.scatter(x,l,label=str(l))
        i+=1
    plt.legend()
    plt.savefig(save_metrics)
