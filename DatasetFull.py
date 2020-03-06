import torch
import numpy as np
from os import listdir
import torchvision.transforms as transforms
import os
from PIL import Image
import pandas as pd

transform_train = transforms.Compose([
    transforms.ToTensor()
])
class FlorinDataset():
    def __init__(self, file_root_blur,file_root_sharp,transform_train):


      self.file_root_blur=file_root_blur
      self.file_root_sharp=file_root_sharp
      self.data_input= pd.read_csv(file_root_blur,sep=' , ',header=None).values
      self.data_target= pd.read_csv(file_root_sharp,sep=' , ',header=None).values
      self.transform_train=transform_train



    def __len__(self):
     return len(self.data_input)


    def __getitem__(self, index):
        for i in range(0,5):
            if i==0:
                inputs=Image.open(self.data_input[index][i])
                inputs=transforms.ToTensor()(inputs)
            else:
                input = Image.open(self.data_input[index][i])
                input = transforms.ToTensor()(input)
                inputs = torch.cat((inputs, input), 0)
            if i==2 :
                targets=Image.open(self.data_target[index][i])
                targets=transforms.ToTensor()(targets)

        return inputs,targets


