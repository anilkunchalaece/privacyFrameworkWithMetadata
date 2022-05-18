# This script is used to generate wireframes for given images in the folder
# It is an extention/modification of the PARE demo.py file

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import sys
import cv2
import shutil
import time
import joblib
import argparse
import json
from loguru import logger
import pandas as pd
import numpy as np
import torch


from multiprocessing import Pool

sys.path.append('.')
from lib.pare.core.tester import PARETester
from lib.pare.utils.demo_utils import (
    download_youtube_clip,
    video_to_images,
    images_to_video,
)

# from smplx import SMPL
from lib.pare.models.head.smpl_head import SMPL,SMPLHead
from lib.pare.core import config, constants
from lib.pare.utils import geometry


CFG = 'data/pare/checkpoints/pare_w_3dpw_config.yaml'
CKPT = 'data/pare/checkpoints/pare_w_3dpw_checkpoint.ckpt'
MIN_NUM_FRAMES = 0

class WireframeGen:
    def __init__(self,args):
        self.args = args
        self.out_img_width = 432 # x_scaled
        self.out_img_height = 240 # y_scaled
        self.in_img_width = 1920 # x_org
        self.in_img_height = 1080 # y_org
        self.draw_bbox_resized = False

        # Copied from PARE demo.py file
        # args.tracker_batch_size = 1
        # input_image_folder = args.src_imgs
        # output_path = os.path.join(args.output_folder, input_image_folder.rstrip('/').split('/')[-1] + '_' + args.exp)
        # os.makedirs(output_path, exist_ok=True)

        # output_img_folder = os.path.join(output_path, 'pare_results')
        # os.makedirs(output_img_folder, exist_ok=True)

        # num_frames = len(os.listdir(input_image_folder))
        # self.tester = PARETester(self.args)        

    def loadDetections(self,fileName):
        # load and format detections as per PARE requirements
        # bbox detections are already generated when generating the metadata
        # here we use those bbox info and resize it 432x240 (Width x Height)
        try :
            with open(fileName) as fd :
                data = json.load(fd)
                df = pd.DataFrame(data["tracklets"])
                df["iNo"] = df.loc[:,"imageName"].apply(lambda x: int(x.split('.')[0]))
                df = df.loc[df["score"] >= 0.90]
                img_shape = data["image_shape"]
                
                dets = {}

                # get bbox and frames per tid
                tids = df["tid"].unique()
                for t in tids :
                    df_t = df.loc[df["tid"]==t]
                    # df_t = df_t.loc[df_t["score"] >= 0.90]
                    # df_t["iNo"] = df.loc[:,"imageName"].apply(lambda x: int(x.split('.')[0]))
                    df_t = df_t.sort_values(by=["iNo"])
                    # print(df_t["bbox"].to_numpy())
                    # print(self.resizeBbox(df_t["bbox"].to_list()))
                    dets[t-1] = {
                            "bbox" : self.resizeBbox(df_t["bbox"].to_list()),
                            "frames" : df_t["iNo"].to_list()
                        }               
                
                return dets

        except Exception as e:
            print(F"unable to load file, failed with exception {e}")
            raise
    
    def resizeBbox(self,bboxes):
        #TODO -> convert these into configurable
        # out_img_width = 432 # x_scaled
        # out_img_height = 240 # y_scaled
        # in_img_width = 1920 # x_org
        # in_img_height = 1080 # y_org
        
        d=[] # scaled bboxes
        scaled_bboxes = []
        
        x_scale = self.out_img_width/self.in_img_width
        y_scale = self.out_img_height/self.in_img_height

        for bbox in bboxes :
            bbox[0] = bbox[0]*x_scale
            bbox[1] = bbox[1]*x_scale 
            bbox[2] = bbox[2]*y_scale
            bbox[3] = bbox[3]*y_scale
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            c_x, c_y = bbox[0] + w / 2, bbox[1] + h / 2
            w = h = np.where(w / h > 1, w, h)
            scaled_bboxes.append([c_x, c_y, w, h])
        return np.array(scaled_bboxes)

    def resizeImgsInDir(self,imagesDir=None):
        if imagesDir != None :
            self.args.src_imgs = imagesDir # if imagesDir specified use it instead of src_imgs in args

        allImgs = os.listdir(self.args.src_imgs)

        with Pool() as pool:
            pool.map(self.resizeImg, allImgs)

    
    def resizeImg(self,img) :
        imgPath = os.path.join(self.args.src_imgs,os.path.basename(img))
        srcImg = cv2.imread(imgPath)
        resizedImg = cv2.resize(srcImg,(self.out_img_width,self.out_img_height),cv2.INTER_CUBIC)

        outImgPath = os.path.join(self.args.tmp_dir,"src","orig_images_scaled",os.path.basename(img))
        # print(outImgPath)
        cv2.imwrite(outImgPath,resizedImg)


    # use PARE to generate wireframes
    def generateWireframes(self,srcImgs, personDetectionFile=None,outDir=None):

        if outDir == None :
            # Create outDir
            outDir = os.path.join(self.args.tmp_dir,"pred")
            os.makedirs(outDir,exist_ok=True)
        
        if personDetectionFile == None :
            personDetectionFile = os.path.join(self.args.tmp_dir,"personDetectionResults.json")

        # check image size in srcImgs
        # if size does not match resize all the images 
        img = cv2.imread(os.path.join(srcImgs,os.listdir(srcImgs)[0]))
        height, width , _ = img.shape

        if (height != self.out_img_height or width == self.out_img_width) :
            dirToStoreResizedImgs = os.path.join(self.args.tmp_dir,"src","orig_images_scaled")
            os.makedirs(dirToStoreResizedImgs,exist_ok=True)
            self.resizeImgsInDir(srcImgs)
            srcImgs = dirToStoreResizedImgs

        dets = self.loadDetections(personDetectionFile)
        # print(dets)
        # sys.exit()
        # draw bbox for resized imgs just to check
        if self.draw_bbox_resized == True :
            bboxDir = os.path.join(self.args.tmp_dir,"bbox")
            os.makedirs(bboxDir,exist_ok=True)

            for d in dets.keys() :
                for idx, f in enumerate(dets[d]["frames"]) : 
                    f = F"{f}.{os.listdir(srcImgs)[0].split('.')[-1]}"
                    tlbr = [int(b) for b in dets[d]["bbox"][idx]]
                    # print(tlbr)
                    if os.path.isfile(os.path.join(bboxDir,f)) :
                        srcImg = cv2.imread(os.path.join(bboxDir,f))
                        # print(srcImg.shape)
                    else :
                        # print(os.path.join(srcImgs,f))
                        srcImg = cv2.imread(os.path.join(srcImgs,f))
                    # srcImg = cv2.imread(os.path.join(srcImgs,f))
                    srcImg = cv2.rectangle(srcImg,(tlbr[0],tlbr[1]),(tlbr[2],tlbr[3]),(255,0,0))
                    srcImg = cv2.putText(srcImg,str(d), (tlbr[0],tlbr[1]), cv2.FONT_HERSHEY_SIMPLEX,0.2,(255,255,0))
                    # srcImg = cv2.putText(srcImg,F"{t.score:0.2f}", (tlbr[2],tlbr[3]), cv2.FONT_HERSHEY_SIMPLEX,0.2,(255,0,0))
                    # print(os.path.join(bboxDir,f))
                    cv2.imwrite(os.path.join(bboxDir,f),srcImg)
                    
        tester = PARETester(self.args)
        # pare_results = tester.run_on_image_folder(srcImgs, dets, outDir)
        pare_results = tester.run_on_video(dets, srcImgs, self.out_img_width, self.out_img_height)
    

        # shape_list = np.zeros(shape=(len(pare_results.keys()),10))
        # for idx, k in enumerate(pare_results.keys()) :
        #     shape_list[idx,:] = np.average(pare_results[k]['betas'],axis=0)        
        # avg_shape = np.average(shape_list,axis=0)

        # normal rendering without changing the shape
        outDirOrig = os.path.join(outDir,"orig")
        os.makedirs(outDirOrig,exist_ok=True)

        tester.render_results(pare_results, srcImgs, outDirOrig,
                                  self.out_img_width, self.out_img_height, len(os.listdir(srcImgs)))

        # render with unified shape
        outDirUnifiedShape = os.path.join(outDir,"unifiedShape")
        os.makedirs(outDirUnifiedShape,exist_ok=True)

        pare_results_uniform_shape = self.customShape(pare_results)
        tester.render_results(pare_results_uniform_shape, srcImgs, outDirUnifiedShape,
                                  self.out_img_width, self.out_img_height, len(os.listdir(srcImgs)))


    def customShape(self,pare_results):
        # Calculate average shape
        shape_list = np.zeros(shape=(len(pare_results.keys()),10))
        for idx, k in enumerate(pare_results.keys()) :
            shape_list[idx,:] = np.average(pare_results[k]['betas'],axis=0)        
        avg_shape = np.average(shape_list,axis=0)

        # print(pare_results[1].keys())
        # print(pare_results[k]['betas'])

        # generate vertices with average shape
        smpl = SMPLHead(config.SMPL_MODEL_DIR)
        for idx, k in enumerate(pare_results.keys()):
            # avg_shape_n = np.tile(avg_shape, (pare_results[k]['betas'].shape[0], 1))
            # print(pare_results[k]['pose'].shape)
            # in PARE pose represented rotation matix representation i.e 24x3x3
            # we need to convert it axis angle representation
            # ref - https://github.com/mkocabas/PARE/issues/4
            # out = smpl(betas=avg_shape_n[1,:],body_pose=geometry.batch_rot2aa(torch.from_numpy(pare_results[k]['pose'][1,:,:,:])))
            # print(out.verts.shape)

            avg_shape_n = torch.from_numpy(np.tile(avg_shape, (pare_results[k]['betas'].shape[0], 1)).astype(np.float32))
            body_pose = torch.from_numpy(pare_results[k]['pose'].astype(np.float32))

            out = smpl(rotmat=body_pose, shape=avg_shape_n)

            pare_results[k]['verts'] = out["smpl_vertices"]

            # for i, f in enumerate(pare_results[k]['pose']) :
            #     betas = torch.from_numpy(avg_shape.astype(np.float32)).reshape(1,-1)
            #     body_pose = geometry.batch_rot2aa(torch.from_numpy(f.astype(np.float32)))[1:,:].reshape(1,-1)
            #     global_orient = geometry.batch_rot2aa(torch.from_numpy(f.astype(np.float32)))[1,:].reshape(1,-1)
            #     trans = torch.from_numpy(pare_results[k]['pred_cam'][i].astype(np.float32))
            #     body_pose = torch.from_numpy(f.astype(np.float32)).reshape(1,24,3,3)

            #     # out = smpl(betas=betas,body_pose=body_pose,global_orient=global_orient,trans=trans)
            #     out = smpl(rotmat=body_pose, shape=betas)
            #     # print(out)
            #     pare_results[k]['verts'][i,:] = torch.squeeze(out["smpl_vertices"])
        return pare_results
        
            


if __name__ == "__main__" :
    args = None
    wg = WireframeGen(args)
    # dets = wg.loadDetections("tmp/personDetectionResults.json")
    # print(dets[1])
    wg.customShape()
            


