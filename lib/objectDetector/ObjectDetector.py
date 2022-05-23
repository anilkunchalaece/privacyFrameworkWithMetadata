import torch
import torchvision
import cv2
import numpy as np 
from torchvision.transforms import transforms
from torchvision.utils import draw_segmentation_masks
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

        # self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # replacing the fastercnn with maskrcnn to get the masks and object detections
        self.detector = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.detector = self.detector.to(self.device)
        self.detector = self.detector.eval()

        self.score_threshold = 0.1 # threshold used for mask generation
        self.batch_size = 2 # batchsize used for running inference

        self.transform = transforms.Compose([transforms.ToTensor()])

    @torch.no_grad()
    def getObjects(self,imageDir,dirToSave):
        dataset = ObjectDataset(imageDir, self.transform)
        dataLoader = DataLoader(dataset,batch_size=self.batch_size)

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
        
        del self.detector

        return outFileName

    # save the masks and return the detection
    @torch.no_grad()
    def saveMasksAndGetObjects(self,imageDir,dirToSave):
        dataset = ObjectDataset(imageDir, self.transform)
        dataLoader = DataLoader(dataset,batch_size=self.batch_size)

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

                    # save the mask
                    # check scores with >= score_threshold
                    scoreIdxs = np.argwhere(det["scores"].detach().cpu().numpy() >= self.score_threshold).squeeze()
                    
                    # based on threshold_score_idxs get label indexes which are only human
                    labelIdxs = np.argwhere(det["labels"][scoreIdxs].detach().cpu().numpy() == 1).squeeze() # label "1" is person TODO -> need to improve this part of code

                    #get masks based on labelIdx
                    masks = det["masks"][labelIdxs].detach().cpu().numpy().squeeze()
                    masks = (masks >= 0.5)
                    #create a tensor with mask and convert it to bool -> suitable for draw_segementation_masks
                    masks = torch.from_numpy(masks).type(torch.bool)

                    dirToSaveMasks = os.path.join(dirToSave,"masks")
                    os.makedirs(dirToSaveMasks,exist_ok=True)

                    maskOutFilePath = os.path.join(dirToSaveMasks,os.path.basename(img_names[i]))
                    backgroundImg = torch.zeros(data[0].shape,dtype=torch.uint8)
                    maskImg = draw_segmentation_masks(backgroundImg, masks,colors=[(255,255,255)]*200).numpy().transpose(1,2,0)
                    cv2.imwrite(maskOutFilePath, cv2.cvtColor(maskImg, cv2.COLOR_RGB2BGR) )

        allDetections["img_shape"] = data[0].shape

        outFileName = os.path.join(dirToSave,"objectDetection.pkl")

        # print(len(allDetections["boxes"]))
        with open(outFileName,'wb') as fw:
            pickle.dump(allDetections,fw)
        
        del self.detector
        torch.cuda.empty_cache()

        return outFileName



if __name__ == "__main__":
    import sys
    sys.path.append("/home/akunchala/Documents/PhDStuff/privacyFrameworkWithMetadata")
    from lib.datasets.objectDataset import ObjectDataset    
    srcDir = "/home/akunchala/Documents/PhDStuff/PrivacyFramework/tmp_icSense_v2_0_left/src/orig_images_scaled"
    dirToSave = "/home/akunchala/Documents/PhDStuff/privacyFrameworkWithMetadata/tmp"
    od = ObjectDetectorClass()
    od.getObjects(srcDir,dirToSave)