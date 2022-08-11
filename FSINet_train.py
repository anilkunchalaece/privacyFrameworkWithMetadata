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
import argparse

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
torch.manual_seed(42)
logger.info(F"running with {device}")

img_width = 64
img_height = 128
batchSize = 50
N_EPOCH = 50

def train(args):
    transform = transforms.Compose([
                                    transforms.Resize((img_height,img_width)),# height,width
                                    transforms.RandomRotation(degrees=45),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    
    pr = PreprocessRAPv2(args.anon_file,args.src_imgs)    
    triplets = pr.generateTriplets()
    # train, valid = train_test_split(triplets,shuffle=True)
    train , valid, test = triplets["train"] , triplets["val"], triplets["test"]

    train_dataset = FusedDataset(args.src_imgs,train, transform)
    valid_dataset = FusedDataset(args.src_imgs,valid, transform)

    train_dataloader = DataLoader(train_dataset,batch_size=FUSED_CONFIG["batchSize"],shuffle=True,drop_last=True)
    valid_dataloader = DataLoader(valid_dataset,batch_size=FUSED_CONFIG["batchSize"]    ,shuffle=True,drop_last=True)

    model = FusedSimilarityNet(FUSED_CONFIG)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(),lr=FUSED_CONFIG["lr"])
    criterion = nn.TripletMarginLoss(margin=FUSED_CONFIG["margin"])

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

        if FUSED_CONFIG["FUSED"] == True :
            mName = F'FSINet_fused_{FUSED_CONFIG["FSI_TYPE"]}' 
        else :
            mName = F'FSINet_image_only{FUSED_CONFIG["FSI_TYPE"]}'

        # earlyStopping
        if _vl < validLossPrev : # if there is a decrease in validLoss all is well 
            badEpoch = 0 # reset bad epochs
            #save model
            torch.save(model.state_dict(),F"models/{mName}e{epoch}.pth")

        else :
            if _vl - validLossPrev >= 0.000001 : # min delta
                badEpoch = badEpoch + 1

            if badEpoch >= patience :
                print(F"Training stopped early due to overfitting in epoch {epoch}")
                break
        validLossPrev = _vl # store current valid loss

    # dump losses into file
    lossFileName = F"{mName}_losses.json"
    with open(lossFileName,"w") as fd:
        json.dump(lossDict,fd)
    print(F"Training and validation losses are saved in {lossFileName}")


def plotLoss():
    fileName = "FSINet_fused_CONCAT_ATTN_losses.json"
    try :
        with open(fileName) as fd:
            d = json.load(fd)
            NO_OF_ITEMS = 1000
            plt.plot(d["train"][:NO_OF_ITEMS],label="Training")
            plt.plot(d["valid"][:NO_OF_ITEMS],label="Validation")
            plt.title("FSINet ( wireframes with org labels) Training loss")
            plt.legend()
            plt.show()
    except Exception as e :
        print(F"unable to open {fileName} with exception {str(e)}")




if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_imgs",type=str,help="input imgs location",required=True, default=None)
    parser.add_argument("--anon_file",type=str,help="annotation file location", required=True, default=None)
    parser.add_argument("--func", type=str, help="function to execute", required=True,default="plot")
    args = parser.parse_args()

    if args.func == "train" :
        train(args)
    elif args.func == "plot" :
        plotLoss()
