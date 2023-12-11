from ast import Import
import time
import random
import numpy as np
import pandas
import torch
import torch.nn as nn
from model import UNet
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from my_dataset import MyDataset
from loss import MyLoss
import os
import albumentations as A
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix ,roc_curve,roc_auc_score
from albumentations.pytorch import ToTensorV2
import hiddenlayer as hl
from PIL import Image
import pandas as pd
import cv2

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
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),

    A.RandomBrightnessContrast(p=0.2),  
    A.Normalize(mean=[0.18], std=[0.24]),
    ToTensorV2()
])

    valid_transform = A.Compose([
        A.Resize(128, 128),
        A.Normalize(mean=[0.18], std=[0.24]),
        ToTensorV2(),
    ])   
    
    start = time.time() 
    kf=KFold(n_splits=k,shuffle=True,random_state=1).split(indexes)
    folder_acc=[]   
    LR=0.0001
    
    inf_DICE=[[[] for i in range(MAX_EPOCH)],[[] for i in range(MAX_EPOCH)]]
    
    
    for x,index  in enumerate(kf):
        train_index, test_index=index
        net = UNet()
        net.to(device)
        net.train()
        

        criterion = MyLoss()
        optimizer = optim.Adam(net.parameters(), weight_decay=weight_decay, lr=LR)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=MAX_EPOCH,
                                                        eta_min=0,
                                                        last_epoch=-1)

        print('\nTraining start!\n')

        h = hl.History()
        c = hl.Canvas()
        


        train_data = MyDataset(data_dir, labels_dir,train_index, transform=train_transform)
        valid_data = MyDataset(data_dir, labels_dir,test_index, transform=valid_transform)
        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)    
  
        max_dice = 0.  
        reached = 0   
        LR_list=[] 
        for epoch in range(1, MAX_EPOCH + 1):   

            DICE_train=[]
            DICE_test=[]
            loss_train = 0.  
            loss_test = 0
            
      
  
            for i, data in tqdm(enumerate(train_loader)):
                               
                inputs, labels = data             
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                optimizer.zero_grad()               
                proba=torch.sigmoid(outputs).cpu().detach().numpy()
                real=labels.cpu().numpy()
                intersect=real*proba
                
                for j in range(len(real)):
                    if epoch==MAX_EPOCH and i==0 and x ==0:
                        
                        prediction=torch.sigmoid(outputs[j][0]).cpu().detach().numpy()
                        label_image=labels[j][0].cpu().numpy()*255
                        print(max(labels[1])) 
                        for f in range(len(prediction)):
                            for s in range(len(prediction[0])):
                                if prediction[f][s]>0.5:
                                    prediction[f][s]=255
                                else:
                                    prediction[f][s]=0   
               
                        cv2.imwrite('output image {}.png'.format(j),prediction)
                        cv2.imwrite('label image {}.png'.format(j),label_image)
                    N=sum(real[j].reshape(-1))
                    M=sum(proba[j].reshape(-1))
                    cross=sum(intersect[j].reshape(-1))
                    DICE_train.append(2*cross/(M+N))

                loss = criterion(outputs.reshape(-1), labels.reshape(-1))          
                loss.backward()
                optimizer.step()
                LR_list.append(optimizer.param_groups[0]['lr'])
                scheduler.step()
                loss_train += loss.item()*outputs.shape[0]
                
            inf_DICE[0][epoch-1].append(sum(DICE_train)/len(DICE_train))
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} DICE: {:.4f} ".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_train/len(train_index),sum(DICE_train)/len(DICE_train)))
                
            net.eval()
            with torch.no_grad():
                for i, data in tqdm(enumerate(valid_loader)):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
     
                    loss = criterion(outputs, labels)
                    proba=torch.sigmoid(outputs).cpu().detach().numpy()
                    real=labels.cpu().numpy()
                    intersect=real*proba
                    loss_test+=loss.item()*outputs.shape[0]
                    
                    for j in range(len(real)):                        
                        N=sum(real[j].reshape(-1))
                        M=sum(proba[j].reshape(-1))
                        cross=sum(intersect[j].reshape(-1))
                        DICE_test.append(2*cross/(M+N))

                if sum(DICE_test)/len(DICE_test) > max_dice:
                    max_dice = sum(DICE_test)/len(DICE_test)
                    reached = epoch
                inf_DICE[1][epoch-1].append(sum(DICE_test)/len(DICE_test))
            
                print("Testing:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} DICE: {:.4f}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_test/len(test_index),sum(DICE_test)/len(DICE_test)))
                
            h.log(epoch, DICE_train=sum(DICE_train)/len(DICE_train),DICE_test=sum(DICE_test)/len(DICE_test),train_loss=loss_train/len(train_index),test_loss=loss_test/len(test_index))
            with c:
              
                c.draw_plot([h["DICE_train"],h['DICE_test']])
                c.draw_plot([h["train_loss"],h['test_loss']])

        # folder_acc.append(max_acc)
        print('\nTraining for the folder {:0>2} finish, the time consumption of {} epochs is {}s\n'.format(x+1,MAX_EPOCH, round(time.time() - start)))
        print('The max validation DICE is: {:.4f}, reached at epoch {}.\n'.format(max_dice, reached))
        c.save("training_progress of folder of Unet {:0>2}.png".format(x+1))
        plt.cla()
        plt.clf()
        plt.plot(list(range(len(LR_list))), LR_list)
        plt.xlabel("epoch")
        plt.ylabel("lr")
        plt.savefig('Learning rate figure for folder {:0>2}.png'.format(x+1))
        plt.cla()
        plt.clf()
    print(111111111111111111111111111111111111111111111111111111111111111111111111)
    print('\nTraining for all {} folders finish, the time consumption of {} epochs is {}s\n'.format(x+1,MAX_EPOCH, round(time.time() - start)))

    
    for i in range(MAX_EPOCH):
     
        inf_DICE[0][i]=sum(inf_DICE[0][i])/k
        inf_DICE[1][i]=sum(inf_DICE[1][i])/k
  
    inf=pd.DataFrame(inf_DICE,columns=np.arange(1,MAX_EPOCH+1),index=['train','test']).T
    ax=sns.lineplot(data=inf)
    ax.set(xlabel='epoch',ylabel='DICE')
    ax.get_figure().savefig('Final_DICE for Unet.png')
    
    
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\nRunning on:", device)

    if device == 'cuda':
        device_name = torch.cuda.get_device_name()
        print("The device name is:", device_name)
        cap = torch.cuda.get_device_capability(device=None)
        print("The capability of this device is:", cap, '\n')
    
    k=5
    seed = 1
    MAX_EPOCH = 120
    BATCH_SIZE = 10
    weight_decay = 1e-3
    data_dir = "original"
    labels_dir='GT'
    indexes_labels=np.array([os.path.join(data_dir, i ) for i in os.listdir(labels_dir)])
    indexes=np.arange(len(indexes_labels))
    set_seed(seed)
    print('random seed:', seed)
    main()