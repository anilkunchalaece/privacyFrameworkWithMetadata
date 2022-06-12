# Siamese Networks with metadata fusion for metadata and wireframe representation learning
import torch
from torchvision import models
import torch.nn as nn
from loguru import logger

class FusedSimilarityNet(nn.Module):
    def __init__(self,config):
        super(FusedSimilarityNet,self).__init__()
        self.resnet = models.resnet18(pretrained=True)

        # freeze layers of pretrained model
        for param in self.resnet.parameters():
            param.requires_grad = False

        fc_inp = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.Linear(fc_inp, config["RESNET_FC1_OUT"]),
            nn.ReLU(),
            nn.Linear(config["RESNET_FC1_OUT"], config["RESNET_FC2_OUT"])
        )

        self.gender_embed_layer = nn.Embedding(config["GENDER_NUM_EMBED"], config["GENDER_EMBED_DIM"])
        self.age_embed_layer = nn.Embedding(config["AGE_NUM_EMBED"], config["AGE_EMBED_DIM"])
        self.body_shape_layer = nn.Embedding(config["BODY_SHAPE_NUM_EMBED"], config["BODY_SHAPE_EMBED_DIM"])
        self.attachment_embed_layer = nn.Embedding(config["ATTACHMENT_NUM_EMBED"],config["ATTACHMENT_EMBED_DIM"])
        self.upper_body_clothing_layer = nn.Embedding(config["UPPER_BODY_CLOTHING_NUM_EMBED"], config["UPPER_BODY_CLOTHING_EMBED_DIM"])
        self.lower_body_clothing_layer = nn.Embedding(config["LOWER_BODY_CLOTHING_NUM_EMBED"], config["LOWER_BODY_CLOTHING_EMBED_DIM"])

        embed_fc_inp = config["GENDER_EMBED_DIM"] + config["AGE_EMBED_DIM"] + config["BODY_SHAPE_EMBED_DIM"] + config["ATTACHMENT_EMBED_DIM"] + config["UPPER_BODY_CLOTHING_EMBED_DIM"] + config["LOWER_BODY_CLOTHING_EMBED_DIM"]

        self.embed_fc = nn.Sequential(
            nn.Linear(embed_fc_inp, config["EMBED_FC1_OUT"]),
            nn.ReLU(),
            nn.Linear(config["EMBED_FC1_OUT"], config["EMBED_FC2_OUT"])
        )


    # x is a dict with following - image and attr
    # attr is a dict of following embeddings 
    # gender, age, bodyShape, attachment, lowerBodyClothing, upperBodyClothing 
    def forward_one(self,x):

        # process image and get its features
        if list(x["imageVal"].shape) == 3 : # if received single image add extra dimension
            x["imageVal"] = x["imageVal"].unsqueeze(0)
        
        # logger.info(F'{x["imageVal"].shape},{x["attrIdxs"][3].unsqueeze(0).shape},{x["attrIdxs"][3]}')
        img_f = self.resnet(x["imageVal"])

        # process attributes and get its features
        embed_base = torch.cat(
                                [
                                    self.gender_embed_layer(x["attrIdxs"][0].unsqueeze(0)), 
                                    self.age_embed_layer(x["attrIdxs"][1].unsqueeze(0)), 
                                    self.body_shape_layer(x["attrIdxs"][2].unsqueeze(0)),
                                    self.attachment_embed_layer(x["attrIdxs"][3].unsqueeze(0)),
                                    self.upper_body_clothing_layer(x["attrIdxs"][4].unsqueeze(0)), 
                                    self.lower_body_clothing_layer(x["attrIdxs"][5].unsqueeze(0))
                                ],dim=2).squeeze()

        embed_f = self.embed_fc(embed_base)
        out = torch.cat([img_f,embed_f],dim=1)
        # logger.info(F"{img_f.shape}, {embed_f.shape},{out.shape}")
        # return {
        #     "img_f" : img_f,  
        #     "embed_f" : embed_f
        # }
        return out

    def forward(self,anchor,positive,negative=None):
        p_out = self.forward_one(positive)
        a_out = self.forward_one(anchor)
        if negative == None :
            return a_out, p_out
        else :
            n_out = self.forward_one(negative)
            return a_out, p_out, n_out





if __name__ == "__main__" :
    import sys
    sys.path.append("/home/akunchala/Documents/PhDStuff/privacyFrameworkWithMetadata")
    from lib.utils.fusedConfig import FUSED_CONFIG
    fSINet = FusedSimilarityNet(FUSED_CONFIG) 
    inp = {
        "image" : torch.rand((2,3,120,60)),
        "gender" : torch.tensor([1,0]),
        "age" : torch.tensor([2,4]),
        "bodyShape" : torch.tensor([1,3]),
        "attachment" : torch.tensor([3,6]),
        "upperBodyClothing" : torch.tensor([10,20]),
        "lowerBodyClothing" : torch.tensor([5,15])
    }

    x = fSINet(inp)
    for k in x.keys() :
        print(x[k].shape)