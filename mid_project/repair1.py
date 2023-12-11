import tkinter as tk
from tkinter import filedialog,messagebox
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import os
from tqdm import tqdm
    

def predict_file(selectFile):

    train_transform = A.Compose([
    A.Resize(128,128),
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    # A.OneOf([
    #     A.IAAAdditiveGaussianNoise(),   # 将高斯噪声添加到输入图像
    #     A.GaussNoise(),    # 将高斯噪声应用于输入图像。
    # ], p=0.2),   # 应用选定变换的概率
    # A.OneOf([
    #     A.MotionBlur(p=0.2),   # 使用随机大小的内核将运动模糊应用于输入图像。
    #     A.MedianBlur(blur_limit=3, p=0.1),    # 中值滤波
    #     A.Blur(blur_limit=3, p=0.1),   # 使用随机大小的内核模糊输入图像。
    # ], p=0.2),
    # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    # 随机应用仿射变换：平移，缩放和旋转输入
    # A.RandomBrightnessContrast(p=0.2),   # 随机明亮对比度
    A.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24]),
    ToTensorV2()
])
    img=cv2.cvtColor(cv2.imread(selectFile), cv2.COLOR_BGR2RGB)
    img=train_transform(image=img)['image']
    img=img.view(1,3,128,128)
    labels_dict=['benign','malignant','normal']
    net = torch.load('net.pkl')  # 加载文件net.pkl, 将其内容赋值给output
    net.eval()
    outputs = net(img)#.reshape(1, 3)
    p = nn.Softmax(dim=1)(outputs)#.unsqueeze(dim=1)
    predicted = torch.argmax(outputs)
    
    return labels_dict[predicted],float(p[0][predicted])
labels_dict=['benign','malignant','normal']
data_dir = "data_code\Dataset_BUSI"
data_info=[os.path.join(data_dir, i ) for i in os.listdir(data_dir)]
wrongdict={}

for i in tqdm(data_info):
    for v in labels_dict:
        if v in i:
            label= v
        predicted,probability = predict_file(i)
        if label != predicted and probability >0.7:
            wrongdict[i]=(label,predicted)

print (wrongdict)


        
