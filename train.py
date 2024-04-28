# -*- coding: utf-8 -*-
# @Time    : 2023/2/20 11:32
# @Author  : Xiao Li
# @File    : train.py
import os
import sys
import torch
import argparse
import logging
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import albumentations as albu
from torchsummary import summary
from ptflops import get_model_complexity_info
#from thop import profile
#from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose, OneOf
import segmentation_models_pytorch as smp
#from segmentation_models_pytorch.encoders import get_preprocessing_fn
from dataset import *
from utils import *

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train_valid/predict", default="train_valid")
    parse.add_argument("--epoch", type=int, default=None)
    parse.add_argument('--arch', type=str,
                       help='Unet/Unet++/MAnet/PSPNet/DeeplabV3+')
    parse.add_argument('--encoder', '-e', type=str,
                       help='resnet34/resnet50/resnet101/xception/vgg19/inceptionv4/mit_b0/mit_b2/mit_b3')
    parse.add_argument("--batch_size", type=int, default=None)
    parse.add_argument('--dataset',
                       help='dsb2018Cell/PanNuke/HCS')
    parse.add_argument('--data_path', help='/data01/lixiao/pre/dsb2018Cell')
    # parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--model_path",  help="load saved model")
    parse.add_argument("--attention", type=str, default=None, help='scse')
    #parse.add_argument("--img_h", type=int, default=256)
    #parse.add_argument("--img_w", type=int, default=256)
    args = parse.parse_args()
    return args

def getLog(args):
    dirname = os.path.join(args.log_dir,args.arch,args.encoder,str(args.dataset),str(args.batch_size),str(args.epoch))
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging


def getModel(args):
    if args.arch == 'Unet':
        model = smp.Unet(
            encoder_name = args.encoder,
            encoder_weights = 'imagenet',
            in_channels = 3,
            classes = 1,
            decoder_attention_type = args.attention,
            activation = 'sigmoid'
        )
    if args.arch == 'Unet++':
        model = smp.UnetPlusPlus(
            encoder_name = args.encoder,
            encoder_weights = 'imagenet',
            in_channels = 3,
            classes = 1,
            decoder_attention_type = args.attention,
            activation='sigmoid'
        )
    if args.arch == 'MAnet':
        model = smp.MAnet(
            encoder_name=args.encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )
    if args.arch == 'PSPNet':
        model = smp.PSPNet(
            encoder_name=args.encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )
    if args.arch == 'DeeplabV3+':
        model = smp.DeepLabV3Plus(
            encoder_name=args.encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation='sigmoid'
            )
    if args.arch == 'FPN':
        model = smp.FPN(
            encoder_name=args.encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation='sigmoid'
            )
    return model

def getDataset(args):
    train_loaders, val_loaders, test_loaders = None, None, None

    if args.action == 'train_valid':
        root = args.data_path

        # print(root, len(img_paths), len(mask_paths))

        if args.dataset == 'dsb2018Cell':
            img_paths = glob(root + '/images/*')
            mask_paths = glob(root + '/masks/*')
            train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
                train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
        if args.dataset == 'PanNuke':
            img_paths = glob(root + '/images/*')
            mask_paths = glob(root + '/masks/*')
            train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
                train_test_split(img_paths, mask_paths, test_size=0.1, random_state=41)
        if args.dataset == 'BBBC039':
            img_paths = glob(root + '/images/*')
            mask_paths = glob(root + '/masks/*')
            train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
                train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
        if args.dataset == 'Mydata':
            img_paths = glob(root + '/images/*')
            mask_paths = glob(root + '/masks/*')
            train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
                train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
        if args.dataset == 'HCS':
            img_paths = glob(root + '/images/*')
            mask_paths = glob(root + '/masks/*')
            train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
                train_test_split(img_paths, mask_paths, test_size=0.1, random_state=41)

        print('train_dataset')
        train_dataset = myDataset(train_img_paths, train_mask_paths, augmentation=train_augmentation,
                                  x_transform=x_transforms, target_transform=y_transforms)
        train_loaders = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        valid_dataset = myDataset(val_img_paths, val_mask_paths, augmentation=None, x_transform=x_transforms,
                                  target_transform=y_transforms)
        val_loaders = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        test_loaders = val_loaders
        # print('train&valid: ', len(train_dataset), len(train_loaders), len(valid_dataset), len(val_loaders))
        print('train_loader, val_loader ', train_loaders.__len__(), val_loaders.__len__())
    elif args.action == 'predict':
        root = args.data_path
        img_paths = glob(root + '/images/*')
        test_dataset = myDataset(img_paths, img_paths, augmentation=None, x_transform=x_transforms,
                                 target_transform=y_transforms)
        test_loaders = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print('test_loader ', test_loaders.__len__())

    return train_loaders, val_loaders, test_loaders


def val(model,best_iou,val_loaders):
    model= model.eval()
    with torch.no_grad():
        i=0   
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_loaders)  
        #print(num)
        for pic, mask,pic_path,mask_path in val_loaders:
            x = pic.to(device)
            y = model(x)

            img_y = torch.squeeze(y).cpu().numpy()  

            hd_total += get_hd(mask_path[0], img_y)
            miou_total += get_iou(mask_path[0],img_y)  
            dice_total += get_dice(mask_path[0],img_y)
            if i < num:i+=1   
        aver_iou = miou_total / num
        aver_hd = hd_total / num
        aver_dice = dice_total/num
        print('Miou=%f,aver_hd=%f,aver_dice=%f' % (aver_iou,aver_hd,aver_dice))
        logging.info('Miou=%f,aver_hd=%f,aver_dice=%f' % (aver_iou,aver_hd,aver_dice))
        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_iou
            print('===========>save best model!')
            if not args.attention:
                torch.save(model.state_dict(), r'./saved_model/'+str(args.arch)+'_'+str(args.encoder)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth')
            elif args.attention:
                torch.save(model.state_dict(), r'./saved_model/' + str(args.arch) + '_' + str(args.encoder) + '_' + str(
                    args.batch_size) + '_' + str(args.dataset) + '_' + str(args.epoch) + '_' + str(args.attention) + '.pth')
        return best_iou,aver_iou,aver_dice,aver_hd

def train(model, criterion, optimizer, train_loader,val_loader, args):
    best_iou,aver_iou,aver_dice,aver_hd = 0,0,0,0
    num_epochs = args.epoch
    loss_list = []
    iou_list = []
    dice_list = []
    hd_list = []

    writer = SummaryWriter('logs')
    for epoch in range(num_epochs):
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_loader.dataset)
        epoch_loss = 0
        step = 0
        for pic, mask, pic_path, mask_path in train_loader:
            writer.add_images("Epoch: {} image".format(epoch), pic, step)
            writer.add_images("Epoch: {} mask".format(epoch), mask, step)
            step += 1
            inputs = pic.to(device)
            labels = mask.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_loader.batch_size + 1, loss.item()))
            logging.info("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_loader.batch_size + 1, loss.item()))

            #writer.add_scalar('Epoch %d train/loss' % (epoch), loss, step)
        loss_list.append(epoch_loss)

        writer.add_scalar("Epoch loss", epoch_loss, epoch)

        best_iou,aver_iou,aver_dice,aver_hd = val(model,best_iou,val_loader)
        iou_list.append(aver_iou)
        dice_list.append(aver_dice)
        hd_list.append(aver_hd)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))

        writer.add_scalar('IOU', aver_iou, epoch)
        writer.add_scalar('dice', aver_dice, epoch)
        writer.add_scalar('hd', aver_hd, epoch)

    writer.close()
    loss_plot(args, loss_list)
    metrics_plot(args, 'iou&dice',iou_list, dice_list)
    metrics_plot(args,'hd',hd_list)
    return model

def test(test_loaders,save_predict=True):
    logging.info('final test........')
    if save_predict ==True:
        dir = os.path.join(r'./saved_predict',str(args.arch),str(args.encoder),str(args.batch_size),str(args.epoch),str(args.dataset))
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')
    if not args.attention:
        model.load_state_dict(torch.load(r'./saved_model/'+str(args.arch)+'_'+str(args.encoder)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth', map_location='cpu'))  # 载入训练好的模型
    elif args.attention:
        model.load_state_dict(torch.load(
            r'./saved_model/' + str(args.arch) + '_' + str(args.encoder) + '_' + str(args.batch_size) + '_' + str(
                args.dataset) + '_' + str(args.epoch) + '_' + str(args.attention) + '.pth', map_location='cpu'))
    model.eval()

    
    with torch.no_grad(): 
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(test_loaders) 
        for pic,_,pic_path,mask_path in test_loaders:
            print('pic_path: ', pic_path[0])
            pic = pic.to(device)
            predict = model(pic)
            predict = torch.squeeze(predict).cpu().numpy()  
            
            iou = get_iou(mask_path[0],predict)
            miou_total += iou  #获取当前预测图的miou，并加到总miou中
            hd_total += get_hd(mask_path[0], predict)
            dice = get_dice(mask_path[0],predict)
            dice_total += dice
            print('iou={},dice={}'.format(iou,dice))

            '''
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('input')
            plt.imshow(Image.open(pic_path[0]))
            print(pic_path[0])
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title('predict')
            plt.imshow(predict,cmap='Greys_r')
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('mask')
            plt.imshow(Image.open(mask_path[0]), cmap='Greys_r')
            '''
            #plt.figure()
            #plt.imshow(predict,cmap='Greys_r')

            
            if save_predict == True:
                #print(mask_path[0])

                img = Image.fromarray(predict*255)
                img = img.convert('L')
                
                saved_pth = dir +'/'+ mask_path[0].split('/')[-1]
                print('saved_path: ', saved_pth)
                img.save(saved_pth)
                #plt.savefig(dir +'/'+ mask_path[0].split('/')[-1])
            #plt.pause(0.01)
            
            #plt.close()
            #if i < num:i+=1   
        #plt.show()
        print('Miou=%f,aver_hd=%f,dice=%f' % (miou_total/num,hd_total/num,dice_total/num))
        logging.info('Miou=%f,aver_hd=%f,dice=%f' % (miou_total/num,hd_total/num,dice_total/num))
        #print('M_dice=%f' % (dice_total / num))

def predict(test_loaders):
    dir = os.path.join(str(args.data_path), str('predict'))
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        print('dir already exist!')
    model.load_state_dict(torch.load(args.model_path , map_location='cpu')) 
    model.eval()

    with torch.no_grad():
        for pic, _, pic_path, mask_path in test_loaders:
            pic = pic.to(device)
            predict = model(pic)
            predict = torch.squeeze(predict).cpu().numpy()  
            # img_y = torch.squeeze(y).cpu().numpy()  
            # print(mask_path[0])
            #print(predict)
            img = Image.fromarray(predict*255)
            img = img.convert('L')

            saved_pth = dir + '/' + mask_path[0].split('/')[-1]
            print('pic_path: ', pic_path[0], '   saved_path: ', saved_pth)
            img.save(saved_pth)
    return


'''
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """进行图像预处理操作
    Args:
        preprocessing_fn (callbale): 数据规范化的函数
            (针对每种预训练的神经网络)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
'''


class DiceFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.dice_loss = smp.losses.DiceLoss(mode='binary')
        self.focal_loss = smp.losses.FocalLoss(mode='binary', alpha=alpha, gamma=gamma)

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)

        focal_loss = self.focal_loss(inputs, targets)

        loss = dice_loss + focal_loss

        return loss


if __name__ == '__main__':
    args = getArgs()

    print(torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    logging = getLog(args)
    print('**************************')
    print('models:%s,\nencoder:%s, \nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.encoder, args.epoch, args.batch_size, args.dataset))
    logging.info('\n=======\nmodels:%s,\nencoder:%s, \nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
                 (args.arch, args.encoder, args.epoch, args.batch_size, args.dataset))
    print('**************************')

    train_augmentation = Compose([
        albu.RandomRotate90(),
        albu.Flip(),
        OneOf([
            albu.HueSaturationValue(),
            #albu.RandomBrightness(),
            #albu.RandomContrast(),
            albu.RandomBrightnessContrast(),
            #albu.VerticalFlip(),
            #albu.Transpose(),
        ], p=1),
        #albu.Resize(512, 512)
        #albu.Normalize(),
        #ToTensorV2(),
    ])

    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        #albu.Normalize(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])  # ->[-1,1]
    ])

    # mask只需要转换为tensor
    y_transforms = transforms.ToTensor()

    #preprocessing_input = get_preprocessing_fn(args.encoder, 'imagenet')

    model = getModel(args).to(device)
    #summary(model, input_size=(3,256,256))
    print('******************')
    flops, params = get_model_complexity_info(model, (3,256,256))
    print('Flops:  ' + flops)
    print('Params: ' + params)

    train_loaders, val_loaders, test_loaders = getDataset(args)
    

    #criterion = torch.nn.BCELoss()
    #criterion = smp.losses.DiceLoss(mode='binary')
    criterion = DiceFocalLoss()
    optimizer = optim.Adam(model.parameters())

    if 'train' in args.action:
        train(model, criterion, optimizer, train_loaders, val_loaders, args)
        test(test_loaders, save_predict=True)
    #if 'test' in args.action:
        #test(test_dataloaders, save_predict=True)
    if args.action == 'predict':
        predict(test_loaders)