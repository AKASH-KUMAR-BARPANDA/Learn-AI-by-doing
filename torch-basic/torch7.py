import torch
from torch.utils.data import DataLoader,Dataset
import numpy as np

wine_dataset = np.loadtxt(
    "/Users/akash/Documents/neural Nine/wine.csv",
    delimiter=',',
    skiprows=1,
    dtype=np.float32
    )



class Data_loading(Dataset):

    def __init__(self,transform= None):

        self.x_data = wine_dataset[:,1:]
        self.y_data = wine_dataset[:,[0]]
        self.n_sample = wine_dataset.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample =   self.x_data[index] ,self.y_data[index]

        if self.transform:
            sample = self.transform(sample)


    def __len__(self):
        return self.n_sample

class ToTensor:
    def __call__(self, sample):
        inputs ,targets  = sample
        return torch.from_numpy(inputs) , torch.from_numpy(targets)



datasets = Data_loading(transform=ToTensor())
data = DataLoader(datasets,batch_size= 32, shuffle= True)
