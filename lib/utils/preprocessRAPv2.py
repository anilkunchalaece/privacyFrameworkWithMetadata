import os
import sys
import numpy as np 
import random
from scipy.io import loadmat
import json
import numpy_indexed as npi
import pandas as pd
from loguru import logger

def getAttrOfIntrest():
    return [
            # Gender
            'Femal',

            # Age
            'AgeLess16','Age17-30','Age31-45','Age46-60','AgeBiger60',
            
            # BodyShape
            'BodyFatter','BodyFat','BodyNormal','BodyThin','BodyThiner',
            
            # Upper body clothing
            'ub-Shirt','ub-Sweater','ub-Vest','ub-TShirt','ub-Cotton','ub-Jacket','ub-SuitUp',
            'ub-Tight','ub-ShortSleeve','ub-Others','ub-ColorBlack','ub-ColorWhite','ub-ColorGray',
            'up-ColorRed','ub-ColorGreen','ub-ColorBlue','ub-ColorSilver','ub-ColorYellow',
            'ub-ColorBrown','ub-ColorPurple','ub-ColorPink','ub-ColorOrange','ub-ColorMixture','ub-ColorOther',
            
            # Lower body clothing
            'lb-LongTrousers','lb-Shorts','lb-Skirt','lb-ShortSkirt','lb-LongSkirt','lb-Dress','lb-Jeans',
            'lb-TightTrousers','lb-ColorBlack','lb-ColorWhite','lb-ColorGray','lb-ColorRed','lb-ColorGreen',
            'lb-ColorBlue','lb-ColorSilver','lb-ColorYellow','lb-ColorBrown','lb-ColorPurple',
            'lb-ColorPink','lb-ColorOrange','lb-ColorMixture','lb-ColorOther',
            
            # Attachment
            'attachment-Backpack','attachment-ShoulderBag','attachment-HandBag','attachment-WaistBag',
            'attachment-Box','attachment-PlasticBag','attachment-PaperBag','attachment-HandTrunk',
            'attachment-Baby','attachment-Other',
        ]


class PreprocessRAPv2:
    # fileLocation -> attribute file location
    # rootDir -> dir where the images are
    def __init__(self,fileLocation,rootDir,MIN_IMGS_IN_TRACKLET=6):
        self.fileLocation = fileLocation
        self.rootDir = rootDir
        self.MIN_IMGS_IN_TRACKLET = MIN_IMGS_IN_TRACKLET
        self.GENDER_ATTR = ['Female', 'Male']
        self.AGE_ATTR = ['AgeLess16','Age17-30','Age31-45','Age46-60','AgeBiger60','NA']
        self.BODY_SHAPE_ATTR = ['BodyFatter','BodyFat','BodyNormal','BodyThin','BodyThiner',"NA"]
        self.ATTACHMENT_ATTR = ['attachment-Backpack','attachment-ShoulderBag','attachment-HandBag','attachment-WaistBag',
                                'attachment-Box','attachment-PlasticBag','attachment-PaperBag','attachment-HandTrunk',
                                'attachment-Baby','attachment-Other', 'NA']
        self.UPPER_BODY_ATTR = ['ub-Shirt','ub-Sweater','ub-Vest','ub-TShirt','ub-Cotton','ub-Jacket','ub-SuitUp',
                                'ub-Tight','ub-ShortSleeve','ub-Others','ub-ColorBlack','ub-ColorWhite','ub-ColorGray',
                                'up-ColorRed','ub-ColorGreen','ub-ColorBlue','ub-ColorSilver','ub-ColorYellow',
                                'ub-ColorBrown','ub-ColorPurple','ub-ColorPink','ub-ColorOrange','ub-ColorMixture','ub-ColorOther',
                                'NA']
        self.LOWER_BODY_ATTR = ['lb-LongTrousers','lb-Shorts','lb-Skirt','lb-ShortSkirt','lb-LongSkirt','lb-Dress','lb-Jeans',
                                'lb-TightTrousers','lb-ColorBlack','lb-ColorWhite','lb-ColorGray','lb-ColorRed','lb-ColorGreen',
                                'lb-ColorBlue','lb-ColorSilver','lb-ColorYellow','lb-ColorBrown','lb-ColorPurple',
                                'lb-ColorPink','lb-ColorOrange','lb-ColorMixture','lb-ColorOther', 'NA'
                                ]        
    
    def processData(self):
        data = loadmat(open(self.fileLocation,'rb'))
        selected_attributes  = np.array(data["RAP_annotation"][0][0][3][0,:].tolist())
        attribute_names = [list(x)[0].strip() for x in data["RAP_annotation"][0][0][2][:,0].tolist()]
        img_names = [list(x)[0] for x in data["RAP_annotation"][0][0][0][:,0].tolist()]
        attr_values = np.array([list(x) for x in data["RAP_annotation"][0][0][1].tolist()])
        idendities = np.array([x for x in data["RAP_annotation"][0][0][-2][:,0].tolist()])

        partition = {}
        partition["train"] = data["RAP_annotation"][0][0][4][0][1][0][0][0][0].tolist()
        partition["test"] = data["RAP_annotation"][0][0][4][0][1][0][0][1][0].tolist()
        partition["val"] = data["RAP_annotation"][0][0][4][0][1][0][0][2][0].tolist()
        partition["val"].remove(84928) # remove last element

        print(F"Original dataset shape is {attr_values.shape}")
        
        attr_intr = getAttrOfIntrest()
        attr_intr_idxs = []
        for attr in attr_intr :
            for idx, attr_name in enumerate(attribute_names) :
                if attr == attr_name :
                    # print(attr, attr_name, idx)
                    attr_intr_idxs.append(idx)
        
        print(F"out of {attr_values.shape[1]} , {len(attr_intr_idxs)} attributes are considered")

        if not (len(attr_intr) == len(attr_intr_idxs)) :
            raise AssertionError("ERROR :::  attr_intr with length {len(attr_intr)} does not match attr_intr_idxs \
                with lenght {len(attr_intr_idxs)}")
        # print(attr_values[:,attr_intr_idxs].shape)
        dataToSend = {
            "images" : np.array(img_names),
            "attributes" : attr_values[:,attr_intr_idxs],
            "attribute_names" : attr_intr,
            "partition" : partition,
            "identities" : np.array(idendities)
        }

        return dataToSend

    def describe(self):
        cam = []
        trackId = []
        line = []
        # for _img in data["images"].tolist() :
        #     img = _img.split("-")
        #     cam.append(img[0])
        #     trackId.append(img[-3])
        #     line.append(img[-1].split(".")[0])

        data = self.processData()

        # outDir = "/home/akunchala/Documents/z_Datasets/RAP_v2/RAP_annotation/tracklets"
        # os.makedirs(outDir,exist_ok=True)

        trackletImgs = {}

        n_identities = np.unique(data["identities"]).tolist()
        for ni in n_identities :
            if ni == -1 or ni == -2 :
                continue
            
            ni_images = data["images"][np.argwhere(data["identities"] == ni)].tolist()
            # ni_images = _data["images"]
            trackletImgs[ni] = {
                                "images" : data["images"][data["identities"] == ni].tolist(),
                                "attributes_names" : [ self.processAttrs(a) for a in data["attributes"][data["identities"] == ni].tolist()],
                                "attributes": data["attributes"][data["identities"] == ni].tolist() 
                                }
        # for a in trackletImgs[2]["attributes"] :
            # print(a)
        # print(sorted(trackletImgs[2], key= lambda x: int(x[0]["images"].split("-")[-2].replace("frame",""))))
        with open("rapv2_tracklets.json","w") as fd :
            json.dump(trackletImgs,fd)

        out = {}

        for k in trackletImgs.keys():
            for idx, img in enumerate(trackletImgs[k]["images"]):
                if not os.path.exists(os.path.join(self.rootDir,img)):
                    print(F"{os.path.join(self.rootDir,img)} not found")
                    trackletImgs[k]["images"].pop(idx)
                    trackletImgs[k]["attributes_names"].pop(idx)
                    trackletImgs[k]["attributes"].pop(idx)
            out[k] = {
                "images" : trackletImgs[k]["images"],
                "attributes_names" : trackletImgs[k]["attributes_names"],
                "attributes" : trackletImgs[k]["attributes"]
            }

        #sort tracklets based on length and return them
        return out

    def processAttrs(self, attrs) :
        # print(attrs)
        attrNames = getAttrOfIntrest()

        # gender = attrs[0]
        # age = attrs[1:1+5]
        # bodyShape = attrs[6:6+5]
        # upperBodyClothing = attrs[11:11+24]
        # lowerBodyClothing = attrs[35:35+22]
        # attachment = attrs[57:57+10]

        attr_idx = [1,6,11,35,57,68]
        attrs_out = []

        if int(attrs[0]) == 0 :
            attrs_out.append("Female")
        else :
            attrs_out.append("Male")

        for i in range(len(attr_idx[:-1])) :
            try :
                idx = attrs[attr_idx[i]:attr_idx[i+1]].index(1.0)
                attrs_out.append(attrNames[attr_idx[i]:attr_idx[i+1]][idx])
            except ValueError as e :
                attrs_out.append("NA")
        return attrs_out


    def attributeNamesToIndex(self,attrNames) :
        gender_idx = self.GENDER_ATTR.index(attrNames[0])
        age_idx = self.AGE_ATTR.index(attrNames[1])
        bodyShape_idx = self.BODY_SHAPE_ATTR.index(attrNames[2])
        attachment_idx = self.ATTACHMENT_ATTR.index(attrNames[5])
        upperBody_idx = self.UPPER_BODY_ATTR.index(attrNames[3])
        lowerBody_idx = self.LOWER_BODY_ATTR.index(attrNames[4])
        
        return [gender_idx,age_idx,bodyShape_idx,attachment_idx,upperBody_idx,lowerBody_idx]


    # Triplets are generating randomly
    def generateTriplets(self):
        logger.info("loading triplets")
        _tracklets = self.describe()
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

if __name__ == "__main__" :
    fileLocation = "/home/akunchala/Documents/z_Datasets/RAP_v2/RAP_annotation/RAP_annotation.mat"
    rootDir = "/home/akunchala/Documents/z_Datasets/RAP_v2/derived/wireframes"
    rapD = PreprocessRAPv2(fileLocation,rootDir)
    rapD.describe()
    # rapD.generateTriplets()()
