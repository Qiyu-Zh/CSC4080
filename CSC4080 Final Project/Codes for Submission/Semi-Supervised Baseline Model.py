import os
import cv2
import timm
import random
import sklearn
import warnings
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import hiddenlayer as hl
import albumentations as A
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2


os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=UserWarning)
sklearn.set_config(print_changed_only=True)
sns.set_style("white")

class KvasirDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: A.Compose = None) -> None:
        super(KvasirDataset, self).__init__()
        print('Decoding images')
        self.imgs = [cv2.imread(path, 1)[:, :, ::-1] for path in tqdm(df['path'].values)]
        self.labels = df['label'].values
        self.transform = transform

    def __getitem__(self, index): 
        if self.transform is not None:
            img = self.transform(image=self.imgs[index])['image']
        return img, self.labels[index]

    def __len__(self): 
        return len(self.labels)


class WeightedFocalLoss(nn.Module):
    def __init__(self, w = np.array([1.8278,7.9619,3.8983,14.1278]), focusing_param = 2, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(WeightedFocalLoss, self).__init__()
        self.w = torch.from_numpy(w).float().to(device)
        self.focusing_param = focusing_param

    def forward(self, output: Tensor, target: Tensor):
        logpt = - F.cross_entropy(output, target, weight=self.w)
        return -((1 - torch.exp(logpt)) ** self.focusing_param) * logpt

def get_resnet18(num_class: int = 2):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.Tanh(),
        nn.Linear(128, num_class)
    )
    return model

def get_resnet50(num_class: int = 2):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.Tanh(),
        nn.Linear(128, num_class)
    )
    return model

def get_resnet18_relu(num_class: int = 2):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, num_class)
    )
    return model

def get_resnet50_relu(num_class: int = 2):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, num_class)
    )
    return model

def get_swintransformer(num_class: int = 2):
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_class)
    model.head = nn.Sequential(
        nn.Linear(model.head.in_features, 128),
        nn.Tanh(),
        nn.Linear(128, num_class)
    )
    return model

def get_convnext(num_class: int = 2):
    return timm.create_model('convnext_tiny', pretrained=True, num_classes=num_class)

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms = True

def main(arg):
    set_seed(arg.seed)

    train_transforms = A.Compose([
        A.RandomResizedCrop(width=arg.input_size, height=arg.input_size, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.25),


        A.Normalize(p=1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    test_transforms = A.Compose([
        A.Resize(width=arg.input_size, height=arg.input_size, p=1),
        A.Normalize(p=1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    img_df = {'path': [], 'label': []}
    for i_class, label in enumerate(arg.labels):
        file_dir = os.path.join(arg.raw_img_dir, label)
        files = os.listdir(file_dir)
        img_df['path'] += [os.path.join(file_dir, file_name) for file_name in files]
        img_df['label'] += [i_class] * len(files)
    img_df = sklearn.utils.shuffle(pd.DataFrame(img_df)).reset_index(drop=True)

    kf = StratifiedKFold(n_splits=arg.folds, shuffle=True, random_state=arg.seed)
    mean_acc = []
    mean_auc = []

    for fold, (train_i, test_i) in enumerate(kf.split(img_df['path'], img_df['label'])):
        if not arg.is_cross_val and fold == 1:
            break
        
        print('Loading Training Set')
        train_datasets = KvasirDataset(img_df.iloc[train_i], train_transforms)
        print('Loading Test Set')
        test_datasets = KvasirDataset(img_df.iloc[test_i], test_transforms)

        train_loader = DataLoader(train_datasets, batch_size=arg.batch_size, shuffle=True)
        test_loader = DataLoader(test_datasets, batch_size=arg.batch_size, shuffle=False)

        print('Loading Model')
        if arg.model == 'resnet18':
            model = get_resnet18(arg.output_size).to(arg.device)
        elif arg.model == 'swin':
            model = get_swintransformer(arg.output_size).to(arg.device)
        else:
            model = get_resnet50(arg.output_size).to(arg.device)


        criterion = WeightedFocalLoss()


        optimizer = optim.Adam(model.parameters(), lr = arg.lr, weight_decay = arg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = arg.epochs, eta_min = 0, last_epoch = -1)

        print(f'\n Fold {fold} Training Start!\n')

        h = hl.History()
        c = hl.Canvas()

        train_loss_all = []
        train_acc_all = []
        train_auc_all = []

        test_loss_all = []
        test_acc_all = []
        test_auc_all = []

        for epoch in range(arg.epochs):
            print('Epoch {}/{}'.format(epoch + 1, arg.epochs))
            model.train()
            ys, pre_scores, pre_labs = [], [], []
            epoch_loss = 0

            for (b_x, b_y) in tqdm(train_loader):
                ys.append(b_y)
                b_x, b_y = b_x.to(arg.device), b_y.to(arg.device)

                output = model(b_x)
                loss = criterion(output, b_y)

                epoch_loss += loss.item() * b_x.size(0)
                pre_labs.append(torch.argmax(output, 1).cpu().detach())
                pre_scores.append(nn.Softmax(dim=1)(output).cpu().detach())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            ys = torch.cat([torch.stack(ys[:-1]).view(-1, ), ys[-1]])
            pre_scores = torch.cat([torch.stack(pre_scores[:-1]).view(-1, arg.output_size), pre_scores[-1]]).numpy()
            if arg.output_size == 2:
                pre_scores = np.max(pre_scores, axis=1)
            pre_labs = torch.cat([torch.stack(pre_labs[:-1]).view(-1, ), pre_labs[-1]])

            train_loss_all.append(float(epoch_loss) / len(ys))
            train_acc_all.append(int(torch.sum(pre_labs == ys)) / len(ys))
            train_auc_all.append(roc_auc_score(ys.numpy(), pre_scores, multi_class='ovo'))
            print('{} Train Loss: {:.4f} Train Acc: {:.4f} Auc Score: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1], train_auc_all[-1]))

            model.eval()
            ys, pre_scores, pre_labs = [], [], []
            epoch_loss = 0

            with torch.no_grad():
                for (b_x, b_y) in tqdm(test_loader):
                    ys.append(b_y)
                    b_x, b_y = b_x.to(arg.device), b_y.to(arg.device)
                    
                    output = model(b_x)
                    loss = criterion(output, b_y)

                    epoch_loss += loss.item() * b_x.size(0)
                    pre_labs.append(torch.argmax(output, 1).cpu().detach())
                    pre_scores.append(nn.Softmax(dim=1)(output).cpu().detach())

            ys = torch.cat([torch.stack(ys[:-1]).view(-1, ), ys[-1]])
            pre_scores = torch.cat([torch.stack(pre_scores[:-1]).view(-1, arg.output_size), pre_scores[-1]]).numpy()
            if arg.output_size == 2:
                pre_scores = np.max(pre_scores, axis=1)
            pre_labs = torch.cat([torch.stack(pre_labs[:-1]).view(-1, ), pre_labs[-1]])

            test_loss_all.append(float(epoch_loss) / len(ys))
            test_acc_all.append(int(torch.sum(pre_labs == ys)) / len(ys))
            test_auc_all.append(roc_auc_score(ys.numpy(), pre_scores, multi_class='ovo'))
            print('{} Test Loss: {:.4f} Test Acc: {:.4f} Auc Score: {:.4f}'.format(epoch, test_loss_all[-1], test_acc_all[-1], test_auc_all[-1]))

            h.log(
                (epoch),
                train_loss = train_loss_all[-1],
                train_acc = train_acc_all[-1],
                train_auc = train_auc_all[-1],
                test_loss = test_loss_all[-1],
                test_acc = test_acc_all[-1],
                test_auc = test_auc_all[-1],
            )
            with c:
                c.draw_plot([h['train_loss'], h['test_loss']])
                c.draw_plot([h['train_acc'], h['test_acc']])
                c.draw_plot([h['train_auc'], h['test_auc']])

        os.makedirs(arg.figure_save_dir, exist_ok=True)
        os.makedirs(arg.model_dir, exist_ok=True)
        
        mean_acc.append(np.array(test_acc_all[-arg.n_last_mean:]).mean())
        mean_auc.append(np.array(test_auc_all[-arg.n_last_mean:]).mean())

        h.save(os.path.join(arg.model_dir, f'fold_{fold}_log_file.pkl'))
        torch.save(model, os.path.join(arg.model_dir, f'fold_{fold}_model.pkl'))

        plt.savefig(os.path.join(arg.figure_save_dir, f'Fold {fold} Training Process.png'))
        plt.savefig(os.path.join(arg.figure_save_dir, f'Fold {fold} Training Process.eps'), format='eps')

    pd.DataFrame({'Acc': mean_acc, 'Auc': mean_auc}).to_csv(os.path.join(arg.model_dir, f'Results.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--raw_img_dir', type = str, default = f'Data')
    parser.add_argument('--train_test_ratio', type = float, default = 0.8)
    parser.add_argument('--is_cross_val', type = bool, default = True)
    parser.add_argument('--folds', type = int, default = 5)

    parser.add_argument('--model_dir', type = str, default = f'Model\\Resnet18')
    parser.add_argument('--figure_save_dir', type = str, default = f'Figures\\Resnet18')
    parser.add_argument('--model', type = str, default = 'resnet18')
    parser.add_argument('--input_size', type = int, default = 128)
    parser.add_argument('--output_size', type = int, default = 4)
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--workers', type = int, default = 0)
    parser.add_argument('--device', default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--epochs', type = int, default = 30)
    parser.add_argument('--loss', default = 'FocalLoss')
    # parser.add_argument('--w', type = torch.Tensor, default = torch.tensor([1, 2], dtype=torch.float))
    parser.add_argument('--lr', type = float, default = 0.0001)
    parser.add_argument('--weight_decay', type = float, default = 0.01)

    parser.add_argument('--labels', type = list, default = os.listdir(f'Data'))
    parser.add_argument('--n_last_mean', type = int, default = 10)

    args = parser.parse_args([])
    main(arg=args)
