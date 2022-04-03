from torch.utils.data import Dataset
import torch
from PIL import Image
import os


class ObjectDataset(Dataset):
    def __init__(self,rootDir,transform):
        self.rootDir = rootDir
        self.transform = transform

        self.allImages = os.listdir(self.rootDir)

    
    def __len__(self):
        return len(self.allImages)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]
        
        img_name = os.path.join(self.rootDir, self.allImages[idx])
        img = Image.open(img_name).convert('RGB')
        img = self.transform(img)

        return {
            "fileName" : img_name,
            "value" : img
        }