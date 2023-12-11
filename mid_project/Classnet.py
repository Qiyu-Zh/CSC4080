import time
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from my_dataset2 import MyDataset
from loss import MyLoss
import os
from timm.models import convnext
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix ,roc_curve,roc_auc_score
import pandas as pd
import timm
import hiddenlayer as hl
import albumentations as A
from albumentations.pytorch import ToTensorV2


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(): 
    
    train_transform = A.Compose([
        A.Resize(128,128),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.OneOf([
        #     A.IAAAdditiveGaussianNoise(),   # 将高斯噪声添加到输入图像
        #     A.GaussNoise(),    # 将高斯噪声应用于输入图像。
        # ], p=0.2),   # 应用选定变换的概率
        # A.OneOf([
        #     A.MotionBlur(p=0.2),   # 使用随机大小的内核将运动模糊应用于输入图像。
        #     A.MedianBlur(blur_limit=3, p=0.1),    # 中值滤波
        #     A.Blur(blur_limit=3, p=0.1),   # 使用随机大小的内核模糊输入图像。
        # ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # 随机应用仿射变换：平移，缩放和旋转输入
        A.RandomBrightnessContrast(p=0.2),   # 随机明亮对比度
        A.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24]),
        ToTensorV2()
    ])

    LR=0.0001

    net=timm.create_model('convnext_tiny',pretrained=True)
    net.head.fc=nn.Sequential(
        nn.Linear(net.head.fc.in_features, 8), 
        nn.Tanh(),
        nn.Linear(8, 3)
    )

    net.to(device)
    net.train()


    criterion = MyLoss()

    optimizer = optim.Adam(net.parameters(), weight_decay=weight_decay, lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                    T_max=MAX_EPOCH,
                                                    eta_min=0,
                                                    last_epoch=-1)


    train_data = MyDataset(data_dir, indexes, transform=train_transform)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)  
  
    for epoch in range(1, MAX_EPOCH + 1):   
        for  data in tqdm(train_loader):
            inputs, labels = data
            print(inputs.size)
            inputs = inputs.to(device)
            labels = labels.to(device)
        
            outputs = net(inputs)
        
            

            # backward

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()
            scheduler.step()
    torch.save(net, 'net.pkl')

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    if device == 'cuda':
        device_name = torch.cuda.get_device_name()

        cap = torch.cuda.get_device_capability(device=None)
        print("The capability of this device is:", cap, '\n')
    
    # hyper-parameters
    seed = 1
    MAX_EPOCH = 60
    BATCH_SIZE = 32
  
    weight_decay = 1e-3
    data_dir = "data_code\Dataset_BUSI"
    indexes=np.arange(len([os.path.join(data_dir, i ) for i in os.listdir(data_dir)]))
    set_seed(seed)

    main()