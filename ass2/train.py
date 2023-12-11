import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from my_dataset import MyDataset
from model import CNN
from loss import MyLoss
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix ,roc_curve,roc_auc_score
import pandas as pd


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(): 
    
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24])
    ])   
    # ============================ step 1/5 define the model ============================
    
    start = time.time() 
 
    kfold=KFold(n_splits=k,  shuffle=True, random_state=1)
    folder_acc=[]
    
    inf=[[[] for i in range(MAX_EPOCH)],[[] for i in range(MAX_EPOCH)]]
   
    for x,data in enumerate(kfold.split(indexes)):

        #import torchvision.models as modelll
        # net = modelll.resnet18()
        # nnn = net.fc.in_features
        # net.fc = nn.Linear(nnn, 3)
        # net = net.to(device)
        net = CNN(3)
        net.to(device)
        net.train()

        # ============================ step 2/5 define the loss function ====================
        criterion = MyLoss()

        # ============================ step 3/5 define the optimizer ========================
        optimizer = optim.Adam(net.parameters(), weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=MAX_EPOCH,
                                                        eta_min=0,
                                                        last_epoch=-1)

        # ============================ step 4/5 train the model =============================
        print('\nTraining start!\n')

        
        


        train_data = MyDataset(data_dir, indexes[data[0]], transform=train_transform)
        valid_data = MyDataset(data_dir, indexes[data[1]], transform=valid_transform)
        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)    
        max_acc = 0.  
        reached = 0    
        for epoch in range(1, MAX_EPOCH + 1):   
 
            sensitivity = []
            specificity = []
            Pre_list=[] 
            labels_list=[]

            loss_mean = 0.  
  
            for i, data in tqdm(enumerate(train_loader)):

                # forward
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
         
                

                # backward
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()

                # update weights
                optimizer.step()

                # results
                _, predicted = torch.max(outputs.data, 1)
        
        # calculate the accuracy and sensitivity and specity and AUC of this training iteration
        ### Begin your code ###
                
                Proba=list(nn.Softmax(dim=1)(outputs).detach().numpy())
                Pre_list=Pre_list+list(predicted)
                labels_list=labels_list+list(labels)
                loss_mean += loss.item()
              
                train_acc=accuracy_score(labels,predicted)
                try:
                    Auc= roc_auc_score(labels, Proba,multi_class='ovo') 
                    
                    con_mat = confusion_matrix(labels,predicted)##here we define the confusion_matrix
                    for j in range(3):
                        number = np.sum(con_mat[:,:])
                        tp = con_mat[j][j]
                        fn = np.sum(con_mat[j,:]) - tp
                        fp = np.sum(con_mat[:,j]) - tp
                        tn = number - tp - fn - fp
                        sen = tp / (tp + fn)
                        spe = tn / (tn + fp)
                        sensitivity.append(sen)  
                        specificity.append(spe)
        ### End your code ###
        # calculate the accuracy of this training iteration
        # print log
 #
                    if (i+1) % log_interval == 0:
                        loss_mean = loss_mean / log_interval
                        print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} AUC:{:.2}\n Sensitivity of benign:{:.2} Sensitivity of malignant:{:.2} Specificity of benign:{:.2} Specificity of malignant:{:.2}".format(
                            epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, train_acc,Auc,sensitivity[0],sensitivity[1],specificity[0],specificity[1]))
                        loss_mean = 0.
                except:
                    if (i+1) % log_interval == 0:
                        loss_mean = loss_mean / log_interval
                        print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} \n".format(
                            epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, train_acc))
                        loss_mean = 0.
            epoch_acc=accuracy_score(labels_list,Pre_list)
            inf[0][epoch-1].append(epoch_acc)
        # validate the model
            if epoch % val_interval == 0:
                Pre_list=[]
                labels_list=[]

                loss_val = 0.
                net.eval()
                with torch.no_grad():
                    for j, data in tqdm(enumerate(valid_loader)):
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = net(inputs)

                        loss = criterion(outputs, labels)

                        _, predicted = torch.max(outputs.data, 1)

                        loss_val += loss.item()
              

                # calculate the accuracy of the validation predictions
                    ### Begin your code ###
                        Pre_list=Pre_list+list(predicted)
                        labels_list=labels_list+list(labels)
                        loss_mean += loss.item()
                    val_acc=accuracy_score(labels_list,Pre_list)
                    if val_acc > max_acc:
                        max_acc = val_acc
                        reached = epoch
                    inf[1][epoch-1].append(val_acc)
                    print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] folder: {:0>2} Loss: {:.4} Acc:{:.2%} \n".format(
                        epoch, MAX_EPOCH, j+1, len(valid_loader),x+1,loss_val,val_acc))
        folder_acc.append(max_acc)
        print('\nTraining for folder {:0>2} finish, the time consumption of {} epochs is {}s\n'.format(x+1,MAX_EPOCH, round(time.time() - start)))
        print('The max validation accuracy is: {:.2%}, reached at epoch {}.\n'.format(max_acc, reached))
    print(111111111111111111111111111111111111111111111111111111111111111111111111)
    print('\nTraining for all {} folders finish, the time consumption of {} epochs is {}s\n'.format(x+1,MAX_EPOCH, round(time.time() - start)))
    print('The max validation accuracy is: {:.2%}.\n'.format(sum(folder_acc)/k))
    
    for i in range(MAX_EPOCH):
        inf[0][i]=sum(inf[0][i])/k
        inf[1][i]=sum(inf[1][i])/k
  
    inf=pd.DataFrame(inf,columns=np.arange(1,MAX_EPOCH+1),index=['train','test']).T
    ax=sns.lineplot(data=inf)
    ax.set(xlabel='epoch',ylabel='accuracy')
    plt.show()
    


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\nRunning on:", device)

    if device == 'cuda':
        device_name = torch.cuda.get_device_name()
        print("The device name is:", device_name)
        cap = torch.cuda.get_device_capability(device=None)
        print("The capability of this device is:", cap, '\n')
    
    # hyper-parameters
    k=5
    seed = 1
    MAX_EPOCH = 50
    BATCH_SIZE = 32
    LR = 0.001
    weight_decay = 1e-3
    log_interval = 2
    val_interval = 1
    data_dir = "Dataset_BUSI"
    indexes=np.arange(len([os.path.join(data_dir, i ) for i in os.listdir(data_dir)]))
    set_seed(seed)
    print('random seed:', seed)
    main()









