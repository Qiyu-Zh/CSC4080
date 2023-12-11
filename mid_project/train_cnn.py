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
from my_dataset import MyDataset
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
from ImageSave import imageSavePIL
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
 
    stf=StratifiedKFold(n_splits=5,shuffle=True,random_state=1).split(indexes,indexes_labels)
    folder_acc=[]
    
    LR=0.0001
    inf=[[[] for i in range(MAX_EPOCH)],[[] for i in range(MAX_EPOCH)]]
    inf_Auc=[[[] for i in range(MAX_EPOCH)],[[] for i in range(MAX_EPOCH)]]
    
    for x,data in enumerate(stf):

        #import torchvision.models as modelll
        # net = modelll.resnet18()
        # nnn = net.fc.in_features
        # net.fc = nn.Linear(nnn, 3)
        # net = net.to(device)
        net = models.alexnet(pretrained=True)
        
        # print(num_features)
        # print()
        # import time
        # time.sleep(10000)
        net.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 3),
        )
        
         # print(num_features)
        # print()
        # import time
        # time.sleep(10000)
        

        net.to(device)
        net.train()

        # ============================ step 2/5 define the loss function ====================
        criterion = MyLoss()

        # ============================ step 3/5 define the optimizer ========================
        optimizer = optim.Adam(net.parameters(), weight_decay=weight_decay, lr=LR)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=MAX_EPOCH,
                                                        eta_min=0,
                                                        last_epoch=-1)

        # ============================ step 4/5 train the model =============================
        print('\nTraining start!\n')

        h = hl.History()
        c = hl.Canvas()
        


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
            Proba_list=[]

            loss_mean = 0.  
  
            for i, data in tqdm(enumerate(train_loader)):

                # forward
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
         
                imageSavePIL(inputs,str(i)+".png",std=[0.18,0.18,0.18],mean=[0.24,0.24,0.24])

                # backward

                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()

                # update weights
                optimizer.step()
                scheduler.step()
                # results
                _, predicted = torch.max(outputs.data, 1)
        
        # calculate the accuracy and sensitivity and specity and AUC of this training iteration
        ### Begin your code ###
                
                Proba_list=Proba_list+list(nn.Softmax(dim=1)(outputs).detach().numpy())
                Pre_list=Pre_list+list(predicted)
                labels_list=labels_list+list(labels)
                loss_mean += loss.item()*len(predicted)
               
            train_acc=accuracy_score(labels_list,Pre_list)
            
            Auc= roc_auc_score(labels_list, Proba_list,multi_class='ovo') 
            loss_mean = loss_mean/len(Pre_list)
            con_mat = confusion_matrix(labels_list,Pre_list)##here we define the confusion_matrix
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
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} AUC:{:.2}\n Sensitivity of benign:{:.2} Sensitivity of malignant:{:.2} Specificity of benign:{:.2} Specificity of malignant:{:.2}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, train_acc,Auc,sensitivity[0],sensitivity[1],specificity[0],specificity[1]))
                
            inf[0][epoch-1].append(train_acc)
            inf_Auc[0][epoch-1].append(train_acc)
        # validate the model

            Pre_list=[]
            labels_list=[]
            Proba_list=[]
            specificity=[]
            sensitivity=[]

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

                    loss_val += loss.item()*len(predicted)
            

            # calculate the accuracy of the validation predictions
                ### Begin your code ###
                    Pre_list=Pre_list+list(predicted)
                    labels_list=labels_list+list(labels)
                    Proba_list=Proba_list+list(nn.Softmax(dim=1)(outputs).numpy())
                val_acc=accuracy_score(labels_list,Pre_list)
                val_Auc= roc_auc_score(labels_list, Proba_list,multi_class='ovo') 
            
                con_mat = confusion_matrix(labels_list,Pre_list)##here we define the confusion_matrix
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
                if val_acc > max_acc:
                    max_acc = val_acc
                    reached = epoch
                inf[1][epoch-1].append(val_acc)
                inf_Auc[1][epoch-1].append(val_Auc)
                loss_val=loss_val/len(Pre_list)
                print("Testing:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} AUC:{:.2}\n Sensitivity of benign:{:.2} Sensitivity of malignant:{:.2} Specificity of benign:{:.2} Specificity of malignant:{:.2}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_val/len(Pre_list), val_acc,Auc,sensitivity[0],sensitivity[1],specificity[0],specificity[1]))
            h.log(epoch, train_acc=train_acc, test_acc=val_acc,train_Auc=Auc,test_Auc=val_Auc,train_loss=loss_mean,test_loss=loss_val)
            with c:
                c.draw_plot([h["train_acc"], h["test_acc"]])
                c.draw_plot([h["train_Auc"],h['test_Auc']])
                c.draw_plot([h["train_loss"],h['test_loss']])

        folder_acc.append(max_acc)
        print('\nTraining for folder {:0>2} finish, the time consumption of {} epochs is {}s\n'.format(x+1,MAX_EPOCH, round(time.time() - start)))
        print('The max validation accuracy is: {:.2%}, reached at epoch {}.\n'.format(max_acc, reached))
        c.save("training_progress of folder of alexnet {:0>2}.png".format(x+1))
    print(111111111111111111111111111111111111111111111111111111111111111111111111)
    print('\nTraining for all {} folders finish, the time consumption of {} epochs is {}s\n'.format(x+1,MAX_EPOCH, round(time.time() - start)))
    print('The max validation accuracy is: {:.2%}.\n'.format(sum(folder_acc)/k))
    plt.cla()
    plt.clf()
    for i in range(MAX_EPOCH):
        inf[0][i]=sum(inf[0][i])/k
        inf[1][i]=sum(inf[1][i])/k
        inf_Auc[0][i]=sum(inf_Auc[0][i])/k
        inf_Auc[1][i]=sum(inf_Auc[1][i])/k
  
    inf=pd.DataFrame(inf,columns=np.arange(1,MAX_EPOCH+1),index=['train','test']).T
    ax=sns.lineplot(data=inf)
    ax.set(xlabel='epoch',ylabel='accuracy')

    ax.get_figure().savefig('Final_acc for alexnet.png')
    plt.cla()
    plt.clf()
    inf_Auc=pd.DataFrame(inf_Auc,columns=np.arange(1,MAX_EPOCH+1),index=['train','test']).T
    ax=sns.lineplot(data=inf_Auc)
    ax.set(xlabel='epoch',ylabel='Auc')

    ax.get_figure().savefig('Final_Auc for alexnet.png')
    plt.cla()
    plt.clf()
    


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
    MAX_EPOCH = 40
    BATCH_SIZE = 32
  
    weight_decay = 1e-3
    data_dir = "data_code\Dataset_BUSI"
    labels_dict={'benign':0,'malignant':1,'normal':2}
    indexes_labels=np.array([os.path.join(data_dir, i ) for i in os.listdir(data_dir)])
    for i,v in enumerate(indexes_labels):
        for j in labels_dict:
            if j in v:
                indexes_labels[i]=labels_dict[j]
    indexes=np.arange(len(indexes_labels))
    set_seed(seed)
    print('random seed:', seed)
    main()