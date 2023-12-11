import os
from pkgutil import get_data
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
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
        self.data_info2=[cv2.imread(i) for i in self.data_info]
        self.data_info2=[cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in self.data_info2]
        ### End your code ###
        
    
    def __getitem__(self, index): 
        '''
        Get sample-label pair according to index
        '''
        ### Begin your code ###
        labels_dict={'benign':0,'malignant':1,'normal':2}
        image= self.data_info2[index]
        path_img=self.data_info[index]
        #img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(image=image)
        for i in labels_dict:
            if i in path_img:
                return img['image'],labels_dict[i]
  
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

