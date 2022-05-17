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

from multiprocessing import Pool

sys.path.append('.')
from lib.pare.core.tester import PARETester
from lib.pare.utils.demo_utils import (
    download_youtube_clip,
    video_to_images,
    images_to_video,
)

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
        self.draw_bbox_resized = True

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
        tester.render_results(pare_results, srcImgs, outDir, outDir,
                                  self.out_img_width, self.out_img_height, len(os.listdir(srcImgs)))



if __name__ == "__main__" :
    args = None
    wg = WireframeGen(args)
    dets = wg.loadDetections("tmp/personDetectionResults.json")
    print(dets[1])
            


