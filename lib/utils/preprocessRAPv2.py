import os
import sys
import numpy as np 
import random
from scipy.io import loadmat
import numpy_indexed as npi

class PreprocessRAPv2:
    def __init__(self,fileLocation,data_percent=100):
        self.fileLocation = fileLocation

    def getAttrOfIntrest(self):
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
    
    def processData(self):
        data = loadmat(open(self.fileLocation,'rb'))
        selected_attributes  = np.array(data["RAP_annotation"][0][0][3][0,:].tolist())
        attribute_names = [list(x)[0].strip() for x in data["RAP_annotation"][0][0][2][:,0].tolist()]
        img_names = [list(x)[0] for x in data["RAP_annotation"][0][0][0][:,0].tolist()]
        attr_values = np.array([list(x) for x in data["RAP_annotation"][0][0][1].tolist()])

        partition = {}
        partition["train"] = data["RAP_annotation"][0][0][4][0][1][0][0][0][0].tolist()
        partition["test"] = data["RAP_annotation"][0][0][4][0][1][0][0][1][0].tolist()
        partition["val"] = data["RAP_annotation"][0][0][4][0][1][0][0][2][0].tolist()
        partition["val"].remove(84928) # remove last element

        print(F"Original dataset shape is {attr_values.shape}")
        
        attr_intr = self.getAttrOfIntrest()
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
            "partition" : partition
        }

        return dataToSend


if __name__ == "__main__" :
    fileLocation = "/home/akunchala/Documents/z_Datasets/RAP_v2/RAP_annotation/RAP_annotation.mat"
    rapD = PreprocessRAPv2(fileLocation)
    rapD.processData()