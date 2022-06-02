from lib.objectDetector.ObjectDetector import ObjectDetectorClass
from lib.utils.getPersonsFromDetections import *
from lib.byteTracker.byte_tracker import BYTETracker
from lib.utils.preprocessRAPv2 import getAttrOfIntrest

import attrTrain

import os
import argparse
import shutil
import json
import pandas as pd
import cv2
from multiprocessing import Pool
import torch
import re
import copy
from sklearn import preprocessing

from wireframeGen import WireframeGen
from lib.pare.core.tester import PARETester

# Config for PARE
CFG = 'data/pare/checkpoints/pare_w_3dpw_config.yaml'
CKPT = 'data/pare/checkpoints/pare_w_3dpw_checkpoint.ckpt'
MIN_NUM_FRAMES = 0

class MetadataGenerator:
    def __init__(self,args):
        self.args = args
        # create tmp dir
        os.makedirs(args.tmp_dir,exist_ok=True)
        self.bboxTracketRef = []

    def getPersonDetections(self,mars=False):
        objDets = ObjectDetectorClass()
        if self.args.det_file == None :
            if mars == False :
                detFile = objDets.saveMasksAndGetObjects(args.src_imgs, args.tmp_dir)
            else :
                detFile = objDets.getObjects(args.src_imgs, args.tmp_dir)
        else :
            detFile = self.args.det_file
        personDets = getPersonsFromPkl(detFile)
        # print(personDets.keys())
        return personDets


    # run the byteTracker to generate metadata
    # used to get persondetections and tracking results will be used wireframification i.e PARE
    def runByteTracker(self):
        dets = self.getPersonDetections()
        tracker = BYTETracker(args)

        results = []
        for idx, det in enumerate(dets["personDetections"]):
            # ref - https://github.com/ifzhang/ByteTrack/issues/23
            #  img_info is the origin size and img_size is the inference size.
            online_targets = tracker.update(det["detections"], dets["img_shape"], dets["img_shape"])
            
            srcImg = cv2.imread(det['img_name']) if args.draw_bbox == True else None

            for t in online_targets:
                tlwh = t.tlwh
                tlbr = [ int(x) for x in t.tlbr.tolist()]
                tid = t.track_id

                # # save results
                # results.append(
                #     f"{os.path.basename(det['img_name'])},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1"
                # )
                results.append({
                    "imageName" : os.path.basename(det['img_name']),
                    "timgName" : F"t_{t.track_id}-{os.path.basename(det['img_name'])}",
                    "tid" : t.track_id,
                    "bbox" : t.tlbr.tolist(), # in format [x1,y1,x2,y2]
                    "score" : t.score
                })

                if self.args.draw_bbox == True :
                    srcImg = cv2.rectangle(srcImg,(tlbr[0],tlbr[1]),(tlbr[2],tlbr[3]),(255,0,0))
                    srcImg = cv2.putText(srcImg,str(t.track_id), (tlbr[0],tlbr[1]), cv2.FONT_HERSHEY_SIMPLEX,0.2,(255,255,0))
                    srcImg = cv2.putText(srcImg,F"{t.score:0.2f}", (tlbr[2],tlbr[3]), cv2.FONT_HERSHEY_SIMPLEX,0.2,(255,0,0))
            
            if self.args.draw_bbox == True:
                outDir = os.path.join(self.args.tmp_dir,"person_bbox")
                os.makedirs(outDir,exist_ok=True) #createDir if not exist, if its already there ignore it 
                outFile = os.path.join(outDir,os.path.basename(det['img_name']))
                cv2.imwrite(outFile,srcImg)

        # dump the data into json file
        outFile = os.path.join(self.args.tmp_dir, "personDetectionResults.json")
        detsObj = {
            "image_shape" : dets["img_shape"], # [channels, width, height]
            "tracklets" : results
        }

        with open(outFile,'w') as fd :
            json.dump(detsObj,fd)
        
        # print(F"total no of values are {len(results)}")
        self.extractBboxForTrackers(results)
    
    def cropBbox(self,imgName):
        tId_DF = self.trackerDf.loc[self.trackerDf["imageName"]==imgName]
        bboxes = tId_DF["bbox"].tolist()
        tIds = tId_DF["tid"].tolist()
        scores = tId_DF["score"].tolist()
        # print(bboxes)
        srcImg = cv2.imread(os.path.join(self.args.src_imgs,imgName))
        for i,bbox in enumerate(bboxes):
            if scores[i] > 0.75 :
                bbox = [int(x) for x in bbox]
                cImg = srcImg[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                
                outFileDir = os.path.join(self.args.tmp_dir,"tracklets",str(tIds[i])) 
                os.makedirs(outFileDir,exist_ok=True)
                outFileName = os.path.join(outFileDir,F"t_{tIds[i]}-{os.path.basename(imgName)}")
                try :
                    cv2.imwrite(outFileName,cImg)
                except :
                    # print(F"unable to extract {tIds[i]} from image {imgName}")
                    pass

    def extractBboxForTrackers(self,trackerList):

        self.trackerDf = pd.DataFrame.from_dict(trackerList)
        allImgs = self.trackerDf["imageName"].unique().tolist()
        self.cropBbox(allImgs[0])
        with Pool() as pool:
            pool.map(self.cropBbox, allImgs)


    def generateMetadaForEachImg(self):
        
        # run the byteTracker to generate metadata
        # used to get persondetections and tracking results will be used wireframification i.e PARE
        self.runByteTracker()
        
        # run inference on tracklet bboxes and get the fileName where attrs are stored
        attrFile = attrTrain.infer(args=self.args)

        imgMetadataDir =  os.path.join(self.args.tmp_dir,"metadata")
        os.makedirs(imgMetadataDir,exist_ok=True) # create a dir if not exists

        # get the detection file
        detectionFile = os.path.join(args.tmp_dir,"personDetectionResults.json")

        #process the detection file to get the detections per image
        with open(detectionFile) as fd :
            dets = json.load(fd)
            img_shape = dets["image_shape"]
            dets = pd.DataFrame(dets["tracklets"])
            # print(dets["imageName"].unique())
        
        with open(attrFile,'rb') as fd :
            attrs = pickle.load(fd)
            # print(attrs)
        
        detImgs = dets["imageName"].unique()
        
        # TODO - speedup this with multiprocessing
        for img in detImgs :
            attr_img = []
            # print(img)
            det_i = dets.loc[dets['imageName'] == img]
            for i, row in det_i.iterrows():
                r = row.to_dict()
                try :
                    attr_i = attrs["imgs"].index(r["timgName"])
                    attr_img.append({
                        "bbox" : r["bbox"],
                        "tid" : r["tid"],
                        "attr" : self.processAttrs(attrs["attrs"][attr_i])
                    })
                except Exception as e:
                    print(F"unable to extract data for {r['timgName']}, failed with exception {e}")
                    # raise
            # save the attr_img file in metadata dir
            fName = os.path.join(imgMetadataDir,F"{img.split('.')[0]}.json")
            outObj = {
                "image_shape" : img_shape,
                "metadata" : attr_img
            }
            # print(fName)
            with open(fName,'w') as fd:
                json.dump(outObj, fd)

    def processAttrs(self,attrs) :
        attrs = attrs.tolist()
        attrNames = getAttrOfIntrest()

        # gender = attrs[0]
        # age = attrs[1:1+5]
        # bodyShape = attrs[6:6+5]
        # upperBodyClothing = attrs[11:11+24]
        # lowerBodyClothing = attrs[35:35+22]
        # attachment = attrs[57:57+10]

        attr_idx = [1,6,11,35,57,68]
        attrs_out = []

        if int(attrs[0]) == 0 :
            attrs_out.append("Female")
        else :
            attrs_out.append("Male")

        for i in range(len(attr_idx[:-1])) :
            try :
                idx = attrs[attr_idx[i]:attr_idx[i+1]].index(1.0)
                attrs_out.append(attrNames[attr_idx[i]:attr_idx[i+1]][idx])
            except ValueError as e :
                attrs_out.append("NA")
        return attrs_out

    # used to check metadata for each tracker
    def checkTrackAttrs(self,tId):
        metadataDir = os.path.join(args.tmp_dir,"metadata")
        fList = sorted(os.listdir(metadataDir),key=lambda f: int(re.sub('\D', '', f)))
        for f in fList :
            with open(os.path.join(metadataDir,f)) as fp :
                d = json.load(fp)
                for md in d["metadata"] :
                    if md["tid"] == tId :
                        print(f, ",".join(md["attr"]))


    # used to generate wireframe for given images using PARE - https://github.com/mkocabas/PARE
    def generateWireframes(self,mars=False):
        wg = WireframeGen(self.args)
        if mars == False :
            wg.generateWireframes(self.args.src_imgs)
        else :
            wg.generateWireframesForMARS(self.args.src_imgs)
    
def renameSrcDir(srcDir) :
    # Check fileNames -> files should start with 0, if not rename them
    fList = sorted(os.listdir(srcDir))
    startIdx = int(fList[0].split(".")[0])
    endIdx = int(fList[-1].split(".")[0])

    if startIdx != 0 :
        for f in fList :
            srcName = os.path.join(srcDir,f)
            desName = os.path.join(srcDir,F"{int(f.split('.')[0]) - 1}.{f.split('.')[-1]}")
            shutil.move(srcName,desName)    

def main(args):
    torch.cuda.empty_cache()
    # renameSrcDir(args.src_imgs)
    mg = MetadataGenerator(args)
    mg.generateMetadaForEachImg()
    mg.generateWireframes()
    torch.cuda.empty_cache()

def main_mars(args):
    tester = PARETester(args)
    with open(args.mars_file) as fd :
        data = json.load(fd)
        tmp_d = copy.deepcopy(args.tmp_dir)
        for k in data.keys() :
            for _s in data[k] :
                args.tmp_dir = os.path.join(tmp_d,F"{k}_{_s}")
                src_imgs = "tmp_mars"
                os.makedirs(src_imgs, exist_ok=True)
                args.src_imgs = src_imgs
                for f in data[k][_s] : 
                    print(f)
                    shutil.copy(f,src_imgs)
                torch.cuda.empty_cache()
                # renameSrcDir(args.src_imgs)
                mg = MetadataGenerator(args)
                # mg.generateMetadaForEachImg()
                mg.getPersonDetections(mars=True)
                # mg.runByteTracker()
                mg.generateWireframes(mars=True, tester=tester)
                torch.cuda.empty_cache()
                shutil.rmtree(src_imgs)
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_imgs",type=str,help="input imgs location",required=False, default=None)
    parser.add_argument("--src_video",type=str,help="input video location", required=False, default=None)
    parser.add_argument("--tmp_dir", type=str,help="tmp dir to store intermediate files", required=False, default="tmp")
    parser.add_argument("--draw_bbox", type=bool,default=False)

    parser.add_argument("--det_file", type=str,help="detection file", default=None)

    # args for BYTE Tracker
    parser.add_argument("--track_thresh", type=float, default=0.75, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=5, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, help="test mot20.")
    
    #bbox image height and width
    parser.add_argument("--img_height",type=str, help="image height for training", required=False, default=256)
    parser.add_argument("--img_width",type=str, help="image width for training", required=False, default=192)
    parser.add_argument("--attr_infer_dir", type=str, help="dir contains tracklets for inference",required=False, default="tmp/tracklets")
    parser.add_argument("--attr_trained_model", type=str, help="path to trained model",required=False, default="models/attrnet_ckpt_975.pth")
    parser.add_argument("--attr_batch_size",type=int, help="batch size used for traning", required=False,default=100)

    #PARE configuration
    parser.add_argument('--cfg', type=str, default=CFG, help='config file that defines model hyperparams')
    parser.add_argument('--ckpt', type=str, default=CKPT, help='checkpoint path')
    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of PARE')
    parser.add_argument('--display', action='store_true', help='visualize the results of each step during demo')
    parser.add_argument('--smooth', action='store_true', help='smooth the results to prevent jitter')
    parser.add_argument('--min_cutoff', type=float, default=0.004, help='one euro filter min cutoff, Decreasing the minimum cutoff frequency decreases slow speed jitter')
    parser.add_argument('--beta', type=float, default=1.0, help='one euro filter beta. Increasing the speed coefficient(beta) decreases speed lag.')
    parser.add_argument('--no_render', action='store_true', help='disable final rendering of output video.')
    parser.add_argument('--no_save', action='store_true', help='disable final save of output results.')
    parser.add_argument('--wireframe', action='store_true', help='render all meshes as wireframes.')
    parser.add_argument('--sideview', action='store_true', help='render meshes from alternate viewpoint.')
    parser.add_argument('--draw_keypoints', action='store_true', help='draw 2d keypoints on rendered image.')
    parser.add_argument('--save_obj', action='store_true', help='save results as .obj files.')

    parser.add_argument("--func",required=True, help="use `mars` if generating for MARS else use `mot` ")
    parser.add_argument("--mars_file",required=False, help="file mars_tracklets wise image list is stored")

    args = parser.parse_args()

    if args.func == "mot" :
        main(args)
    elif args.func == "mars":
        main_mars(args)
    else :
        print("please use func argument either with mot or mars")
    # mg = MetadataGenerator(args)
    # mg.runByteTracker()
    # mg.generateMetadaForEachImg()
    # mg.checkTrackAttrs(10)