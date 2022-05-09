from torch.utils.data import Dataset
import torch
from PIL import Image
import os
from torchvision.transforms import transforms


class AttrDataset(Dataset):
    def __init__(self,src_imgs, attrs,transform, src_dir):
        self.imgs = src_imgs,
        self.attrs = attrs
        self.srcDir = src_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.attrs)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]

        imgPath = os.path.join(self.srcDir, self.imgs[0][idx])
        img = Image.open(imgPath).convert('RGB')
        labels = torch.tensor(self.attrs[idx,:], dtype=torch.float)

        return {
            "image" : self.transform(img),
            "labels" : labels
        }

class AttrDatasetInference(Dataset):
    def __init__(self, src_dir,transform):
        self.srcDir = src_dir
        self.transform = transform
        self.all_imgs = self.combineAllTracklets()    

    def __len__(self):
        return len(self.all_imgs)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]

        imgPath = self.all_imgs[idx]
        img = Image.open(imgPath).convert('RGB')

        return {
            "image" : self.transform(img),
            "fileName" : imgPath
        }
    
    def combineAllTracklets(self) :
        img_list = []
        for d in os.listdir(self.srcDir):
            for img in os.listdir(os.path.join(self.srcDir,d)) :
                img_list.append(os.path.join(self.srcDir,d,img))
        # print(len(img_list))
        # print(img_list[:2])
        return img_list