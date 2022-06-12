"""
This script is used to train FSINet 
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision

from sklearn.model_selection import train_test_split

from lib.fsinet.FSINet import FusedSimilarityNet
from lib.utils.fusedConfig import FUSED_CONFIG
from lib.utils.preprocessRAPv2 import *
from lib.datasets.fusedDataset import FusedDataset

import matplotlib.pyplot as plt
import numpy as np
import json
from loguru import logger

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
torch.manual_seed(42)
logger.info(F"running with {device}")

img_width = 60
img_height = 120
batchSize = 100
N_EPOCH = 100

def train():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((img_height,img_width)),# height,width
                                    transforms.RandomRotation(degrees=45)])
    
    annotationFile = "/home/akunchala/Documents/z_Datasets/RAP_v2/RAP_annotation/RAP_annotation.mat"
    rootDir = "/home/akunchala/Documents/z_Datasets/RAP_v2/derived/wireframes"
    pr = PreprocessRAPv2(annotationFile)    
    triplets = pr.generateTriplets()
    train, valid = train_test_split(triplets,shuffle=True)

    train_dataset = FusedDataset(rootDir,train, transform)
    valid_dataset = FusedDataset(rootDir,valid, transform)

    train_dataloader = DataLoader(train_dataset,batch_size=batchSize,shuffle=True,drop_last=True)
    valid_dataloader = DataLoader(valid_dataset,batch_size=batchSize,shuffle=True,drop_last=True)

    model = FusedSimilarityNet(FUSED_CONFIG)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(),lr = 0.00001 )
    criterion = nn.TripletMarginLoss(margin=0.1)

    # earlyStopping params
    patience = 10 # wait for this many epochs before stopping the training
    validLossPrev = float("inf") #used for early stopping
    badEpoch = 0

    # dict to store losses
    lossDict = {
            "train" : [],
            "valid" : []
        }

    for epoch in range(0,N_EPOCH) :
        
        tl = []
        vl = []
        
        # training
        model.train()
        for idx, data in enumerate(train_dataloader):
            # print(data)
            anchorImgs = data["anchor"]#.to(device)
            anchorImgs["imageVal"] = anchorImgs["imageVal"].to(device)
            anchorImgs["attrIdxs"] = [ x.to(device) for x in anchorImgs["attrIdxs"]]
            
            positiveImgs = data["positive"]#.to(device)
            positiveImgs["imageVal"] = positiveImgs["imageVal"].to(device)
            positiveImgs["attrIdxs"] = [ x.to(device) for x in positiveImgs["attrIdxs"]]
            
            negativeImgs = data["negative"]#.to(device)
            negativeImgs["imageVal"] = negativeImgs["imageVal"].to(device)
            negativeImgs["attrIdxs"] = [ x.to(device) for x in negativeImgs["attrIdxs"]]


            opt.zero_grad()
            a_f,p_f,n_f = model(anchorImgs,positiveImgs,negativeImgs)
            loss = criterion(a_f,p_f,n_f)
            loss.backward()
            opt.step()
            tl.append(loss.cpu().detach().item())
        
        # validation
        model.eval()
        for idx, data in enumerate(valid_dataloader):

            anchorImgs = data["anchor"]#.to(device)
            anchorImgs["imageVal"] = anchorImgs["imageVal"].to(device)
            anchorImgs["attrIdxs"] = [ x.to(device) for x in anchorImgs["attrIdxs"]]
            
            positiveImgs = data["positive"]#.to(device)
            positiveImgs["imageVal"] = positiveImgs["imageVal"].to(device)
            positiveImgs["attrIdxs"] = [ x.to(device) for x in positiveImgs["attrIdxs"]]
            
            negativeImgs = data["negative"]#.to(device)
            negativeImgs["imageVal"] = negativeImgs["imageVal"].to(device)
            negativeImgs["attrIdxs"] = [ x.to(device) for x in negativeImgs["attrIdxs"]]

            a_f,p_f,n_f = model(anchorImgs,positiveImgs,negativeImgs)
            loss = criterion(a_f,p_f,n_f)
            vl.append(loss.cpu().detach().item())

        # collect losses
        _tl, _vl = np.mean(tl) , np.mean(vl)
        lossDict["train"].append(_tl)
        lossDict["valid"].append(_vl)
        print(F"epoch:{epoch}, tl:{_tl}, vl:{_vl}")

        # earlyStopping
        if _vl < validLossPrev : # if there is a decrease in validLoss all is well 
            badEpoch = 0 # reset bad epochs

            #save model
            torch.save(model.state_dict(),"models/FSINet.pth")

        else :
            if _vl - validLossPrev >= 0.0001 : # min delta
                badEpoch = badEpoch + 1

            if badEpoch >= patience :
                print(F"Training stopped early due to overfitting in epoch {epoch}")
                break
        validLossPrev = _vl # store current valid loss

    # dump losses into file
    lossFileName = "fsiNet_losses.json"
    with open(lossFileName,"w") as fd:
        json.dump(lossDict,fd)
    print(F"Training and validation losses are saved in {lossFileName}")







if __name__ == "__main__" :
    train()