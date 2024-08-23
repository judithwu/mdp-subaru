from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import torch as torch
import cv2
from util import pad_img
class DrowsyDataset(Dataset):
    def __init__(self, drowsy_data, awake_data):
        self.drowsy_data = drowsy_data
        self.awake_data = awake_data
        self.images = self.create_dataset()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = self.images[index]
        return img
    
    def create_dataset(self):
        images = []
        drowsy_length = len(os.listdir(self.drowsy_data))
        awake_length = len(os.listdir(self.awake_data))
            
        with tqdm(total=drowsy_length+awake_length, unit="img") as pbar:
          for file in os.listdir(self.drowsy_data):
            img = cv2.imread(os.path.join(self.drowsy_data, file))
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = transforms.ToTensor()(img)  
            img = pad_img(img, 60, 80)
            images.append((img, 1))       
            pbar.update()

          for file in os.listdir(self.awake_data):
            img = cv2.imread(os.path.join(self.awake_data, file))
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = transforms.ToTensor()(img)  
            img = pad_img(img, 60, 80)
            images.append((img, 0))       
            pbar.update()  
       
        return images          
    

