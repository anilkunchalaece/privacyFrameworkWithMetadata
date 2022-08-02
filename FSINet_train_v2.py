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
# from lib.utils.fusedConfig import FUSED_CONFIG
from lib.utils.preprocessRAPv2 import *
from lib.datasets.fusedDataset import FusedDataset

import matplotlib.pyplot as plt
import numpy as np
import json
from loguru import logger
import argparse
from functools import partial

#imports for rayTune
from ray import tune
from ray.tune import CLIReporter 
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
torch.manual_seed(42)
logger.info(F"running with {device}")

img_width = 60
img_height = 120
batchSize = 100
N_EPOCH = 10
LIMIT_DATA = True

def train(config,checkpoint_dir=None):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                                    transforms.Resize((img_height,img_width)),# height,width
                                    transforms.RandomRotation(degrees=45)])
    
    pr = PreprocessRAPv2(args.anon_file,args.src_imgs)    
    triplets = pr.generateTriplets()

    if LIMIT_DATA :
        # train, valid = train_test_split(triplets[:2500],shuffle=True)
        train , valid, test = triplets["train"][:1000] , triplets["val"][:200], triplets["test"][:200]
    else :
        train , valid, test = triplets["train"] , triplets["val"], triplets["test"]

    train_dataset = FusedDataset(args.src_imgs,train, transform)
    valid_dataset = FusedDataset(args.src_imgs,valid, transform)

    train_dataloader = DataLoader(train_dataset,batch_size=config["batchSize"],shuffle=True,drop_last=True)
    valid_dataloader = DataLoader(valid_dataset,batch_size=config["batchSize"],shuffle=True,drop_last=True)

    model = FusedSimilarityNet(config)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(),lr = config["lr"] )
    criterion = nn.TripletMarginLoss(margin = config["margin"])

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

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), opt.state_dict()), path)
        
        tune.report(
            loss=_vl,
            tloss=_tl
        )



def plotLoss():
    fileName = "fsiNet_losses.json"
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


def main(args,num_samples=10,max_num_epochs=20, gpus_per_trial=2):
    # config = {
    #     # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    #     # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    #     "lr": 0.005,
    #     "margin" : 0.10,
    #     "batchSize": 100,
    #     "GENDER_EMBED_DIM" : 2,
    #     "AGE_EMBED_DIM" : 4,
    #     "BODY_SHAPE_EMBED_DIM" : 2,
    #     "ATTACHMENT_EMBED_DIM" :4,
    #     "UPPER_BODY_CLOTHING_EMBED_DIM" :  4,
    #     "LOWER_BODY_CLOTHING_EMBED_DIM" :  4,
    #     "GENDER_NUM_EMBED" : 2,
    #     "AGE_NUM_EMBED" : 6,
    #     "BODY_SHAPE_NUM_EMBED" : 6,
    #     "ATTACHMENT_NUM_EMBED" : 11,
    #     "UPPER_BODY_CLOTHING_NUM_EMBED" : 25,
    #     "LOWER_BODY_CLOTHING_NUM_EMBED" : 23,
    #     "EMBED_FC1_OUT" : tune.grid_search([256,512,1024]),
    #     "EMBED_FC2_OUT" : tune.grid_search([256,512,1024]),
    #     "RESNET_FC1_OUT" : tune.grid_search([256,512,1024]),
    #     "RESNET_FC2_OUT" : tune.grid_search([256,512,1024]),
    #     "FC1_OUT" : tune.grid_search([256,512,1024]),
    #     "FC2_OUT" : tune.grid_search([256,512,1024]),
    # }
    config_old = {
        # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "margin" : tune.loguniform(0.1,0.5),
        "batchSize": tune.choice([25, 50, 100, 150]),
        "GENDER_EMBED_DIM" : tune.choice([1,2]),
        "AGE_EMBED_DIM" : tune.choice(range(2,6)),
        "BODY_SHAPE_EMBED_DIM" : tune.choice(range(2,6)),
        "ATTACHMENT_EMBED_DIM" : tune.choice(range(2,6)),
        "UPPER_BODY_CLOTHING_EMBED_DIM" : tune.choice(range(2,6)),
        "LOWER_BODY_CLOTHING_EMBED_DIM" : tune.choice(range(2,6)),
        "GENDER_NUM_EMBED" : 2,
        "AGE_NUM_EMBED" : 6,
        "BODY_SHAPE_NUM_EMBED" : 6,
        "ATTACHMENT_NUM_EMBED" : 11,
        "UPPER_BODY_CLOTHING_NUM_EMBED" : 25,
        "LOWER_BODY_CLOTHING_NUM_EMBED" : 23,
        "EMBED_FC1_OUT" : tune.choice([128,256,512,1024]),
        "EMBED_FC2_OUT" : tune.choice([128,256,512,1024]),
        "RESNET_FC1_OUT" : tune.choice([128,256,512,1024]),
        "RESNET_FC2_OUT" : tune.choice([128,256,512,1024]),
        "FC1_OUT" : tune.choice([128,256,512,1024]),
        "FC2_OUT" : tune.choice([128,256,512,1024]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=2,
        reduction_factor=2)
    
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss","tloss", "training_iteration"],
        print_intermediate_tables=False)

    search_alg = BasicVariantGenerator(random_state=42)
    
    result = tune.run(
        partial(train),
        config=config,
        resources_per_trial={"cpu": 10, "gpu": 1},
        num_samples=num_samples,
        scheduler=scheduler,
        # search_alg=search_alg,
        progress_reporter=reporter,
        local_dir="tune_results")
    
    best_trial = result.get_best_trial("loss", "min", "last")    
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["tloss"]))
    best_checkpoint_value = best_trial.checkpoint.value
    print(F"best checkpoint value {best_checkpoint_value}")

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_imgs",type=str,help="input imgs location",required=True, default=None)
    parser.add_argument("--anon_file",type=str,help="annotation file location", required=True, default=None)
    parser.add_argument("--func", type=str, help="function to execute", required=True,default="plot")
    args = parser.parse_args()

    if args.func == "train" :
        main(args)
    elif args.func == "plot" :
        plotLoss()
