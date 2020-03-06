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
        ps = 256

        H =720
        W =1280

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)


        for i in range(0,5):
            if i==0:
                input_patch=Image.open(self.data_input[index][i])
                input_patch = transforms.ToTensor()(input_patch)
                input_patchs = input_patch[:,yy:yy + ps, xx:xx + ps]

            else:
                input_patch = Image.open(self.data_input[index][i])
                input_patch = transforms.ToTensor()(input_patch)
                input_patch = input_patch[:,yy:yy + ps, xx:xx + ps]

                input_patchs = torch.cat((input_patchs, input_patch), 0)
            if i==2 :
                target_patch=Image.open(self.data_target[index][i])
                target_patch = transforms.ToTensor()(target_patch)
                target_patch = target_patch[:,yy:yy+ ps, xx:xx + ps]


        return input_patchs,target_patch


