# This script is used to train attrNet

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau # Learning rate Scheduler


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np
import json,os,sys,pickle
import matplotlib.pyplot as plt
import argparse

from lib.utils.preprocessRAPv2 import PreprocessRAPv2, getAttrOfIntrest
from lib.datasets.attrDataset import AttrDataset, AttrDatasetInference
from lib.attr.attrNet import AttrNet

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(F"running with device {device}")
torch.manual_seed(42)

def train_loop(dataloader,model,loss_fcn, optimizer):
    model.train()
    
    _tl = [] # array to capture the training loss
    
    for idx, data in enumerate(dataloader):
        img = data["image"].to(device)
        labels = data["labels"].to(device)

        # compute the prediction and loss
        pred_labels = model(img)
        loss = loss_fcn(pred_labels,labels)
        
        _tl.append(loss.cpu().detach().item())

        # back propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.mean(_tl)

def valid_loop(dataloader,model,loss_fcn):
    model.eval()

    _vl = [] # array to capture the validation loss

    for idx, data in enumerate(dataloader):
        img = data["image"].to(device)
        labels = data["labels"].to(device)

        # compute the prediction and loss
        pred_labels = model(img)
        loss = loss_fcn(pred_labels,labels)

        _vl.append(loss.cpu().detach().item())
    
    return np.mean(_vl)



def train(args):
    data = PreprocessRAPv2(args.dataset,args.src_dir).processData()
    
    if args.data_percent != 100:
        data["partition"]["train"] = data["partition"]["train"][:int(len(data["partition"]["train"])/(100-args.data_percent))]
        data["partition"]["test"] = data["partition"]["test"][:int(len(data["partition"]["test"])/(100-args.data_percent))]
        data["partition"]["val"] = data["partition"]["val"][:int(len(data["partition"]["val"])/(100-args.data_percent))]

        print(F'the parition for training is train : {len(data["partition"]["train"])},test : {len(data["partition"]["test"])}, val : {len(data["partition"]["val"])} ')
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_height,args.img_width)),
        transforms.RandomRotation(degrees=45)
    ])

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_height,args.img_width)),
        # transforms.RandomRotation(degrees=45)
    ])
    
    trainDataset = AttrDataset(data["images"][data["partition"]["train"]],data["attributes"][data["partition"]["train"],:],transform_train,args.src_dir)
    valDataset = AttrDataset(data["images"][data["partition"]["val"]].tolist(),data["attributes"][data["partition"]["val"],:],transform_eval,args.src_dir)


    trainDataLoader = DataLoader(trainDataset,batch_size=args.attr_batch_size,shuffle=True,drop_last=True)
    valDataLoader = DataLoader(valDataset,batch_size=args.attr_batch_size,shuffle=True,drop_last=True)
    
    print(F"train dataloader length {len(trainDataLoader)} val dataloader length {len(valDataLoader)}")

    n_attr = data["attributes"].shape[1]
    print(F"No of attributes to build NN is {n_attr}")
    
    model = AttrNet(n_attr)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(),lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Ref - https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/
    # lr = lr * factor 
    # mode='max': look for the maximum validation accuracy to track
    # patience: number of epochs - 1 where loss plateaus before decreasing LR
            # patience = 0, after 1 bad epoch, reduce LR
    # factor = decaying factor
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, verbose=True)

    lossDict = {
        "tl" : [],
        "vl" : []
    }

    # earlyStopping params
    patience = 10 # wait for this many epochs before stopping the training
    validLossPrev = float("inf") #used for early stopping
    badEpoch = 0

    # create a dir to save attrmodels
    os.makedirs(os.path.join(args.tmp_dir,"models","attrNet"),exist_ok=True)

    for epoch in range(args.nepochs):
        tl = train_loop(trainDataLoader, model, criterion, opt)
        vl = valid_loop(valDataLoader, model, criterion)
        
        #run lr scheduler
        scheduler.step(vl)

        # Early stopping
        if vl < validLossPrev :
            badEpoch = 0 # reset bad epoch
            if epoch % 3 == 0 : torch.save(model.state_dict(),F"{args.tmp_dir}/models/attrNet/attrnet_ckpt_{epoch}.pth")
        else :
            if vl - validLossPrev >= 0.0001 : # min_delta
                badEpoch = badEpoch + 1
            if badEpoch >= patience :
                print(F"Training stopped early due to overfitting in epoch {epoch}")
                break
        validLossPrev = vl # store current valid loss

        # save the losses in dict for postprocess
        lossDict["tl"].append(tl)
        lossDict["vl"].append(vl)

    #once training loop terminated, dump the losses into file
    with open(os.path.join(os.path.join(args.tmp_dir,"attrNetLosses.json")),'w') as fd:
        json.dump(lossDict,fd)


def infer(args):
    # data = PreprocessRAPv2(args.dataset).processData()
    # n_attr = data["attributes"].shape[1]
    attrs = getAttrOfIntrest()
    n_attr = len(attrs)

    transform_infer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_height,args.img_width)),
    ])

    
    attr_infer_dir = os.path.join(args.tmp_dir,"tracklets")
    inferDataset = AttrDatasetInference(attr_infer_dir, transform_infer)


    inferDataLoader = DataLoader(inferDataset,batch_size=args.batch_size,shuffle=True,drop_last=False)

    print(F"infer dataloader length {len(inferDataLoader)}")  
    model = AttrNet(n_attr)
    model.load_state_dict(torch.load(args.attr_trained_model))
    model = model.to(device)
    model.eval()

    results = {
        "attrs" : np.empty(shape=(0,n_attr)),
        "attribute_names" : attrs,
        "imgs" : []
    }
    
    for _img in inferDataLoader :
        img = _img["image"].to(device)
        pred_attrs = model(img)
        
        pred_attrs =  torch.sigmoid(pred_attrs)
        pred_attrs = pred_attrs.detach().cpu().numpy()
        pred_attrs[pred_attrs >= 0.50] = 1
        pred_attrs[pred_attrs < 0.50] = 0

        results["attrs"] = np.append(results["attrs"], pred_attrs,axis=0)
        results["imgs"].extend([os.path.basename(i) for i in _img["fileName"]])
    
    # print(results["attrs"].shape)
    outFile = os.path.join(args.tmp_dir,"tracklet_attrs.pickle")

    with open(outFile,"wb") as fd:
        pickle.dump(results,fd)
    
    del model
    
    return outFile

def plotLoss():
    fileName = "FSINet_fused_CONCAT_ATTN_losses.json"
    try :
        with open(fileName) as fd:
            d = json.load(fd)
            NO_OF_ITEMS = 100
            plt.plot(d["train"][:NO_OF_ITEMS],label="Training")
            plt.plot(d["valid"][:NO_OF_ITEMS],label="Validation")
            plt.title("AttrNet training and validataion losses")
            plt.legend()
            plt.show()
    except Exception as e :
        print(F"unable to open {fileName} with exception {str(e)}")


if __name__ == "__main__" :
    # cmdToRun - python attrTrain.py --src_dir /home/akunchala/Documents/z_Datasets/RAP_v2/RAP_dataset --dataset /home/akunchala/Documents/z_Datasets/RAP_v2/RAP_annotation/RAP_annotation.mat
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, help="train / infer / plot",required=False, default="infer")

    # args for traning
    parser.add_argument("--src_dir",type=str,help="input imgs location",required=False, default=None)
    parser.add_argument("--img_height",type=str, help="image height for training", required=False, default=256)
    parser.add_argument("--img_width",type=str, help="image width for training", required=False, default=192)
    parser.add_argument("--nepochs", type=int, help="no of epochs for training", required=False, default=100)
    parser.add_argument("--lr", type=float, help="learning rate for training", required=False, default=0.00001)
    parser.add_argument("--attr_batch_size",type=int, help="batch size used for traning", required=False,default=100)
    parser.add_argument("--data_percent",type=float, help="percentage of data to be used for training", required=False, default=100)
    parser.add_argument("--dataset", type=str, help="dataset/attributes location", required=False)
    parser.add_argument("--tmp_dir", type=str,help="tmp dir to store intermediate files", required=False, default="tmp")
    
    # args for inference
    # parser.add_argument("--attr_infer_dir", type=str, help="dir contains tracklets for inference",required=False, default="tmp/tracklets")
    parser.add_argument("--attr_trained_model", type=str, help="path to trained model",required=False, default="models/attrnet_ckpt_975.pth")

    args = parser.parse_args()

    if args.action == "train" :
        train(args)
    elif args.action == "plot" :
        plotLoss()
    else :
        infer(args)