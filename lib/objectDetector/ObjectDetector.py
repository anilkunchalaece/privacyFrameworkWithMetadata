import torch
import torchvision
import cv2
import numpy as np 
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os,json,shutil
import pandas as pd
import pickle
try :
    from lib.datasets.objectDataset import ObjectDataset 
except :
    pass

class ObjectDetectorClass:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.detector = self.detector.to(self.device)
        self.detector = self.detector.eval()

        self.transform = transforms.Compose([transforms.ToTensor()])


    def getObjects(self,imageDir,dirToSave):
        dataset = ObjectDataset(imageDir, self.transform)
        dataLoader = DataLoader(dataset,batch_size=10)

        allDetections = {
            "boxes" : [],
            "labels" : [],
            "scores" : [],
            "img_names" : []
        }

        for idx, data in enumerate(dataLoader):
            with torch.no_grad():
                img_names = data["fileName"]
                data = data["value"].to(self.device)
                out = self.detector(data)
                
                for i, det in enumerate(out) :
                    allDetections["boxes"].append(det["boxes"].detach().cpu().tolist())
                    allDetections["labels"].append(det["labels"].detach().cpu().tolist())
                    allDetections["scores"].append(det["scores"].detach().cpu().tolist())
                    allDetections["img_names"].append(img_names[i])

        allDetections["img_shape"] = data[0].shape

        outFileName = os.path.join(dirToSave,"objectDetection.pkl")

        # print(len(allDetections["boxes"]))
        with open(outFileName,'wb') as fw:
            pickle.dump(allDetections,fw)
        
        return outFileName




if __name__ == "__main__":
    import sys
    sys.path.append("/home/akunchala/Documents/PhDStuff/privacyFrameworkWithMetadata")
    from lib.datasets.objectDataset import ObjectDataset    
    srcDir = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_icSense_v2_0_left/src/orig_images_scaled"
    dirToSave = "/home/akunchala/Documents/PhDStuff/privacyFrameworkWithMetadata/tmp"
    od = ObjectDetectorClass()
    od.getObjects(srcDir,dirToSave)