# Siamese Networks with metadata fusion for metadata and wireframe representation learning
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class FusedSimilarityNet(nn.Module):
    def __init__(self,config):
        super(FusedSimilarityNet,self).__init__()
        self.resnet = models.efficientnet_b7(pretrained=True)

        self.config = config

        # freeze layers of pretrained model
        for param in self.resnet.parameters():
            param.requires_grad = False
        # print(self.resnet)
        # fc_inp = self.resnet.fc.in_features
        fc_inp = 2560
        # fc_inp = self.resnet.classifier.in_features

        self.resnet.classifier = nn.Sequential(
            nn.Linear(fc_inp, config["RESNET_FC1_OUT"]),
            # nn.BatchNorm1d(config["RESNET_FC1_OUT"]),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(config["RESNET_FC1_OUT"], config["RESNET_FC2_OUT"])
        )
        if config["FUSED"] == True :
                self.gender_embed_layer = nn.Embedding(config["GENDER_NUM_EMBED"], config["GENDER_EMBED_DIM"])
                self.age_embed_layer = nn.Embedding(config["AGE_NUM_EMBED"], config["AGE_EMBED_DIM"])
                self.body_shape_layer = nn.Embedding(config["BODY_SHAPE_NUM_EMBED"], config["BODY_SHAPE_EMBED_DIM"])
                self.attachment_embed_layer = nn.Embedding(config["ATTACHMENT_NUM_EMBED"],config["ATTACHMENT_EMBED_DIM"])
                self.upper_body_clothing_layer = nn.Embedding(config["UPPER_BODY_CLOTHING_NUM_EMBED"], config["UPPER_BODY_CLOTHING_EMBED_DIM"])
                self.lower_body_clothing_layer = nn.Embedding(config["LOWER_BODY_CLOTHING_NUM_EMBED"], config["LOWER_BODY_CLOTHING_EMBED_DIM"])

                embed_fc_inp = config["GENDER_EMBED_DIM"] + config["AGE_EMBED_DIM"] + config["BODY_SHAPE_EMBED_DIM"] + config["ATTACHMENT_EMBED_DIM"] + config["UPPER_BODY_CLOTHING_EMBED_DIM"] + config["LOWER_BODY_CLOTHING_EMBED_DIM"]

                self.embed_fc = nn.Sequential(
                    nn.Linear(embed_fc_inp, config["EMBED_FC1_OUT"]),
                    # nn.BatchNorm1d(config["EMBED_FC1_OUT"]),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(config["EMBED_FC1_OUT"], config["EMBED_FC2_OUT"])
                )

                if config["FSI_TYPE"] == "CONCAT" or config["FSI_TYPE"] == "CONCAT_ATTN" :
                    fsi_in_shape = config["EMBED_FC2_OUT"]+config["RESNET_FC2_OUT"]
                elif config["FSI_TYPE"] == "ADD" :
                    fsi_in_shape = config["EMBED_FC2_OUT"]
                elif config["FSI_TYPE"] == "ATTN_IMG_F" :
                    fsi_in_shape = config["RESNET_FC2_OUT"]

                self.fsi_out = nn.Sequential(
                                            nn.Linear(fsi_in_shape, config["FC1_OUT"]),
                                            # nn.BatchNorm1d(config["FC1_OUT"]),
                                            nn.ReLU(),
                                            nn.Dropout(),
                                            nn.Linear(config["FC1_OUT"],config["FC2_OUT"])
                                            )

                self.img_attn = nn.MultiheadAttention(config["EMBED_FC2_OUT"],config["ATTN_NO_HEADS"],batch_first=True,dropout=0.5)
                self.embed_attn = nn.MultiheadAttention(config["EMBED_FC2_OUT"],config["ATTN_NO_HEADS"],batch_first=True,dropout=0.5)

                self.embed_self_attn = nn.MultiheadAttention(config["EMBED_FC2_OUT"], config["ATTN_NO_HEADS"], batch_first=True, dropout=0.4)
                self.embed_self_attn_2 = nn.MultiheadAttention(config["EMBED_FC2_OUT"], config["ATTN_NO_HEADS"], batch_first=True, dropout=0.4)
            
                print("running fused images")

    # x is a dict with following - image and attr
    # attr is a dict of following embeddings 
    # gender, age, bodyShape, attachment, lowerBodyClothing, upperBodyClothing 
    def forward_one(self,x):

        # process image and get its features
        if list(x["imageVal"].shape) == 3 : # if received single image add extra dimension
            x["imageVal"] = x["imageVal"].unsqueeze(0)
        
        # logger.info(F'{x["imageVal"].shape},{x["attrIdxs"][3].unsqueeze(0).shape},{x["attrIdxs"][3]}')
        img_f = self.resnet(x["imageVal"])

        if self.config["FUSED"] == True :

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
            if self.config["FSI_TYPE"] == "CONCAT_ATTN" : 
                # print(F"img_shape : {img_f.shape}, embed_shape: {embed_f.shape}, {torch.unsqueeze(embed_f,1).shape}")
                # img_f, embed_f = self.attention(img_f, embed_f)
                
                # embed_f,_ = self.embed_self_attn(torch.unsqueeze(embed_f,1),torch.unsqueeze(embed_f,1),torch.unsqueeze(embed_f,1))
                # embed_f = torch.squeeze(embed_f,1)

                img_f_attn_out,_ = self.img_attn(torch.unsqueeze(img_f,1), torch.unsqueeze(embed_f,1), torch.unsqueeze(embed_f,1))
                embed_f_attn_out,_ = self.embed_attn(torch.unsqueeze(embed_f,1), torch.unsqueeze(img_f,1), torch.unsqueeze(img_f,1))
                # print(F"{img_f_attn_out.shape} , {embed_f_attn_out.shape}")

                # img_f_attn_out,_ = self.embed_self_attn_2(embed_f_attn_out,embed_f_attn_out,embed_f_attn_out)

                out = self.fsi_out(torch.cat([img_f_attn_out.squeeze(1),embed_f_attn_out.squeeze(1)],dim=1))
            
            elif self.config["FSI_TYPE"] == "ATTN_IMG_F" :
                embed_f,_ = self.embed_self_attn(torch.unsqueeze(embed_f,1),torch.unsqueeze(embed_f,1),torch.unsqueeze(embed_f,1))
                embed_f = torch.squeeze(embed_f,1)

                img_f_attn_out,_ = self.img_attn(torch.unsqueeze(img_f,1), torch.unsqueeze(embed_f,1), torch.unsqueeze(embed_f,1))
                # embed_f_attn_out,_ = self.embed_attn(torch.unsqueeze(embed_f,1), torch.unsqueeze(img_f,1), torch.unsqueeze(img_f,1))
                # print(F"{img_f_attn_out.shape} , {embed_f_attn_out.shape}")

                # img_f_attn_out,_ = self.embed_self_attn_2(embed_f_attn_out,embed_f_attn_out,embed_f_attn_out)

                # out = self.fsi_out(torch.cat([img_f_attn_out.squeeze(1),embed_f_attn_out.squeeze(1)],dim=1))     
                out = self.fsi_out(img_f_attn_out.squeeze(1))           
                
            elif self.config["FSI_TYPE"] == "CONCAT" :
                out = self.fsi_out(torch.cat([img_f,embed_f],dim=1))

            elif self.config["FSI_TYPE"] == "ADD" :
                out = self.fsi_out(torch.add(embed_f,img_f,alpha=self.config["FSI_ALPHA"]))
            # logger.info(F"{img_f.shape}, {embed_f.shape},{out.shape}")
            # return {
            #     "img_f" : img_f,  
            #     "embed_f" : embed_f
            # }
            # print("sending fused")
            out = out.div(out.norm(p=2,dim=1,keepdim=True))
            return out
        else :
            # print("sending img only")
            return img_f

    def forward(self,anchor,positive,negative=None):
        p_out = self.forward_one(positive)
        a_out = self.forward_one(anchor)
        if negative == None :
            return a_out, p_out
        else :
            n_out = self.forward_one(negative)
            return a_out, p_out, n_out


    def attention(self,img_weights, metadata_weights):
        img_weights_1 = img_weights.unsqueeze(1) # add extra dimesion Bx1XN
        metadata_weights_1 = metadata_weights.unsqueeze(1) # Bx1xN
        
        img_weights_2 = img_weights_1.permute(0,2,1) # BxNx1
        metadata_weights_2 = metadata_weights_1.permute(0,2,1) # BxNx1

        # print(img_weights_2.shape, metadata_weights_1.shape)

        dot_p = metadata_weights_1 @ img_weights_2 # dot product
        dot_p = dot_p.view(-1)

        img_weights_ret = img_weights.transpose(0,1) * dot_p 
        img_weights_ret = F.softmax(img_weights_ret.transpose(1,0))

        metadata_weights_ret = metadata_weights.transpose(0,1) * dot_p
        metadata_weights_ret = F.softmax(metadata_weights_ret.transpose(1,0))

        return img_weights_ret, metadata_weights_ret



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