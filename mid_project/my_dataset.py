import os
from re import I
import torch
import random
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_dir, indexes, transform=None):
        """
        My Dataset for CAD with BUSI dataset
            param data_dir: str, path for the dataset
            param train: whether this is defined for training or testing
            param transform: torch.transform，data pre-processing pipeline
        """
        ### Begin your code ###
        self.data_info = self.get_data(data_dir,indexes)
        self.transform = transform
        self.images=[Image.open(i).convert('RGB') for i in self.data_info]
        
        ### End your code ###
        
    
    def __getitem__(self, index): 
        '''
        Get sample-label pair according to index
        '''
        ### Begin your code ###
        labels_dict={'benign':0,'malignant':1,'normal':2}
        path_img= self.data_info[index]
        
        img = self.images[index]

        if self.transform is not None:
            img = self.transform(img)
        for i in labels_dict:
            if i in path_img:
                return img,labels_dict[i]
  
        ### End your code ###

    def __len__(self): 
        '''return the size of the dataset'''
        ### Begin your code ###
        return len(self.data_info)
        ### End your code ###
        
    @staticmethod
    def get_data(data_dir,indexes):
        '''
        Load the dataset and store it in your own data structure(s)
        '''
        ### Begin your code ###
        data_info=[os.path.join(data_dir, i ) for i in os.listdir(data_dir)]
        
        return [data_info[i] for i in indexes]
        ### End your code ###

