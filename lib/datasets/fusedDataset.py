from torch.utils.data import Dataset
import torch
import random
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from torchvision.transforms import transforms
try :
    from lib.utils.preprocessRAPv2 import *
except :
    pass
from loguru import logger

class FusedDataset(Dataset):
    def __init__(self,rootDir,triplets,transform):
        self.rootDir = rootDir
        self.transform = transform

        self.allTriplets = triplets


    
    def __len__(self):
        return len(self.allTriplets)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]
        
        # img_name = os.path.join(self.rootDir, self.allImages[idx])
        # img = Image.open(img_name).convert('RGB')
        # img = self.transform(img)

        anchor = self.allTriplets[idx]["anchor"]
        positive = self.allTriplets[idx]["positive"]
        negative = self.allTriplets[idx]["negative"]

        anchor["imageVal"] = self.transform(Image.open(os.path.join(self.rootDir,anchor["image"])).convert('RGB'))
        positive["imageVal"] = self.transform(Image.open(os.path.join(self.rootDir,positive["image"])).convert('RGB'))
        negative["imageVal"] = self.transform(Image.open(os.path.join(self.rootDir,negative["image"])).convert('RGB'))

        return {
            "anchor" : anchor,
            "positive" : positive,
            "negative" : negative
        }

    # Triplets are generating randomly
    def generateTriplets(self,annotationFile):
        logger.info("loading triplets")
        pr = PreprocessRAPv2(annotationFile)
        _tracklets = pr.describe()
        # tIds = sorted(tracklets,key=lambda f: len(tracklets[f]["images"]),reverse=True)
        tracklets = [_tracklets[t] for t in _tracklets if len(_tracklets[t]["images"]) >= self.MIN_IMGS_IN_TRACKLET ]

        logger.info(F"org tracklets {len(_tracklets)} , filtered tracklets {len(tracklets)}")

        triplets = []
        totalTriplets = 0
        for idx,t in enumerate(tracklets) :
            _t_other = [x for x in range(len(tracklets)) if x != idx ]
            pairsToGenerate = int(len(t["images"])/2)
            negativeImgIdxes = random.sample(_t_other, pairsToGenerate)

            anchor_range = list(range(0,pairsToGenerate))
            positive_range = list(range(pairsToGenerate,len(t["images"])))
            
            for pIdx in range(pairsToGenerate):
                anchor_idx =  random.choice(anchor_range)
                positive_idx = random.choice(positive_range)
                negative_idx = random.choice(list(range(len(tracklets[negativeImgIdxes[pIdx]]["images"]))))

                anchor = {
                    "image" : t["images"][anchor_idx],
                    "attrIdxs" : self.attributeNamesToIndex(t["attributes_names"][anchor_idx]),
                    "gender" : t["attributes_names"][anchor_idx][0],
                    "age" : t["attributes_names"][anchor_idx][1],
                    "bodyShape" : t["attributes_names"][anchor_idx][2],
                    "attachment" : t["attributes_names"][anchor_idx][5],
                    "upperBodyClothing" : t["attributes_names"][anchor_idx][3],
                    "lowerBodyClothing" : t["attributes_names"][anchor_idx][4]
                }

                positive = {
                    "image" : t["images"][positive_idx],
                    "attrIdxs" : self.attributeNamesToIndex(t["attributes_names"][positive_idx]),
                    "gender" : t["attributes_names"][positive_idx][0],
                    "age" : t["attributes_names"][positive_idx][1],
                    "bodyShape" : t["attributes_names"][positive_idx][2],
                    "attachment" : t["attributes_names"][positive_idx][5],
                    "upperBodyClothing" : t["attributes_names"][positive_idx][3],
                    "lowerBodyClothing" : t["attributes_names"][positive_idx][4]
                }

                negative = {
                    "image" : tracklets[negativeImgIdxes[pIdx]]["images"][negative_idx],
                    "attrIdxs" : self.attributeNamesToIndex(tracklets[negativeImgIdxes[pIdx]]["attributes_names"][negative_idx]),
                    "gender" : tracklets[negativeImgIdxes[pIdx]]["attributes_names"][negative_idx][0],
                    "age" : tracklets[negativeImgIdxes[pIdx]]["attributes_names"][negative_idx][1],
                    "bodyShape" : tracklets[negativeImgIdxes[pIdx]]["attributes_names"][negative_idx][2],
                    "attachment" : tracklets[negativeImgIdxes[pIdx]]["attributes_names"][negative_idx][5],
                    "upperBodyClothing" : tracklets[negativeImgIdxes[pIdx]]["attributes_names"][negative_idx][3],
                    "lowerBodyClothing" : tracklets[negativeImgIdxes[pIdx]]["attributes_names"][negative_idx][4]
                }

                triplets.append({
                    "positive" : positive,
                    "negative" : negative,
                    "anchor" : anchor
                })

        logger.info(F"total no of triplets {len(triplets)}")
        return triplets

    def attributeNamesToIndex(self,attrNames) :
        gender_idx = self.GENDER_ATTR.index(attrNames[0])
        age_idx = self.AGE_ATTR.index(attrNames[1])
        bodyShape_idx = self.BODY_SHAPE_ATTR.index(attrNames[2])
        attachment_idx = self.ATTACHMENT_ATTR.index(attrNames[5])
        upperBody_idx = self.UPPER_BODY_ATTR.index(attrNames[3])
        lowerBody_idx = self.LOWER_BODY_ATTR.index(attrNames[4])
        
        return [gender_idx,age_idx,bodyShape_idx,attachment_idx,upperBody_idx,lowerBody_idx]

    def visualizeTriplets(self) :
        triplets = self.generateTriplets()
        triplets_sel = random.sample(list(range(len(triplets))),9)
        # select 3 triplets and show them in grid
        fig = plt.figure()
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(3, 9),  # creates 2x2 grid of axes
                        axes_pad=0.25,  # pad between axes in inch.
                        )

        imgs = []
        titles = []
        for idx in triplets_sel :
            _imgs = []
            # print(os.path.join(rootDir,triplets[idx]["anchor"]["image"]))
            # print(os.path.join(rootDir,triplets[idx]["positive"]["image"]))
            # print(os.path.join(rootDir,triplets[idx]["negative"]["image"]))
            _imgs.append(cv2.imread(os.path.join(rootDir,triplets[idx]["anchor"]["image"]))[:,:,::-1])
            _imgs.append(cv2.imread(os.path.join(rootDir,triplets[idx]["positive"]["image"]))[:,:,::-1])
            _imgs.append(cv2.imread(os.path.join(rootDir,triplets[idx]["negative"]["image"]))[:,:,::-1])
            imgs.extend(_imgs)
            titles.extend(['Anchor','Positive','Negative'])

        for ax, im, t in zip(grid, imgs,titles):
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            ax.set_title(t)
        plt.show()

if __name__ == "__main__" :
    import sys
    sys.path.append("/home/akunchala/Documents/PhDStuff/privacyFrameworkWithMetadata")
    from lib.utils.preprocessRAPv2 import *
    annotationFile = "/home/akunchala/Documents/z_Datasets/RAP_v2/RAP_annotation/RAP_annotation.mat"
    rootDir = "/home/akunchala/Documents/z_Datasets/RAP_v2/RAP_dataset"

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((120,60)),
        # transforms.RandomRotation(degrees=45)
    ])

    pR = PreprocessRAPv2(annotationFile)
    fd = FusedDataset(rootDir,pR.generateTriplets(),transform_eval)
    print(fd[0].keys())
    



