import pandas as pd
import os
import glob
import json

def getAttributeNames():
    # ref - https://github.com/yuange250/MARS-Attribute
    return [
            "ub-black", "ub-purple", "ub-green", "ub-blue", "ub-gray", "ub-white", "ub-yellow", "ub-red", "ub-complex",
            "lb-white", "lb-purple", "lb-black", "lb-green", "lb-gray", "lb-pink", "lb-yellow", "lb-blue", "lb-brown", "lb-complex",
            "age",
            "top length",
            "bottom length",
            "shoulder bag",
            "backpack",
            "hat",
            "hand bag",
            "hair",	
            "gender",
            "bottom type"
    ]

class PreprocessMars:
    def __init__(self,srcDir):
        self.attrFileLocation = os.path.join(srcDir,"mars_attributes.csv")
        self.srcDir = srcDir

    def preprocess(self):
        data = pd.read_csv(self.fileLocation)
        allPersonIds = data["person_id"].unique()
        
        # based on person_id
        for pId in allPersonIds :
            _d = data[data["person_id"] == pId]
            
            allTracklets = _d["tracklets_id"].unique()
            _imgNames = []
            for tId in allTracklets :
                pass
    
    def getImgList(self):
        testDir = os.path.join(self.srcDir,"bbox_test")
        trainDir = os.path.join(self.srcDir,"bbox_train")

        out = {}

        for _dir in [testDir,trainDir] :    
            for d in os.listdir(os.path.join(_dir)) :
                _cDir = os.path.join(_dir,d)
                _cFiles = os.listdir(_cDir)
                out[d] = {}
                for f in _cFiles :
                    _cTid = f[6:11]
                    if out[d].get(_cTid,None) == None :
                        out[d][_cTid] = [os.path.join(_dir,d,f)]
                    else :
                        out[d][_cTid].append(os.path.join(_dir,d,f))

        with open(os.path.join(self.srcDir,"mars_tracklet_wise_image_list.json"),"w") as fd :
            json.dump(out, fd)
         







if __name__ == "__main__" :
    fLocation = "/home/akunchala/Documents/z_Datasets/MARS_Dataset/data"
    pm = PreprocessMars(fLocation)
    pm.getImgList()