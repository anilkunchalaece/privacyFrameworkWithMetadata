from lib.objectDetector.ObjectDetector import ObjectDetectorClass
from lib.utils.getPersonsFromDetections import *
from lib.byteTracker.byte_tracker import BYTETracker
from lib.utils.preprocessRAPv2 import getAttrOfIntrest

import attrTrain

import argparse
import json
import pandas as pd
import cv2
from multiprocessing import Pool
from sklearn import preprocessing

class MetadataGenerator:
    def __init__(self,args):
        self.args = args
        # create tmp dir
        os.makedirs(args.tmp_dir,exist_ok=True)
        self.bboxTracketRef = []

    def getPersonDetections(self):
        objDets = ObjectDetectorClass()
        if self.args.det_file == None :
            detFile = objDets.getObjects(args.src_imgs, args.tmp_dir)
        else :
            detFile = self.args.det_file
        personDets = getPersonsFromPkl(detFile)
        # print(personDets.keys())
        return personDets
    
    def runByteTracker(self):
        dets = self.getPersonDetections()
        tracker = BYTETracker(args)

        results = []
        for idx, det in enumerate(dets["personDetections"]):
            # ref - https://github.com/ifzhang/ByteTrack/issues/23
            #  img_info is the origin size and img_size is the inference size.
            online_targets = tracker.update(det["detections"], dets["img_shape"], dets["img_shape"])
            
            srcImg = cv2.imread(det['img_name']) if args.draw_bbox == True else None
            # print(online_targets)
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
        with open(outFile,'w') as fd :
            json.dump(results,fd)
        
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
        # run inference on tracklet bboxes and get the fileName where attrs are stored
        attrFile = attrTrain.infer(args=self.args)

        imgMetadataDir =  os.path.join(self.args.tmp_dir,"metadata")
        os.makedirs(imgMetadataDir,exist_ok=True) # create a dir if not exists

        # get the detection file
        detectionFile = os.path.join(args.tmp_dir,"personDetectionResults.json")

        #process the detection file to get the detections per image
        with open(detectionFile) as fd :
            dets = json.load(fd)
            dets = pd.DataFrame(dets)
            # print(dets["imageName"].unique())
        
        with open(attrFile,'rb') as fd :
            attrs = pickle.load(fd)
            # print(attrs)
        
        detImgs = dets["imageName"].unique()
        
        # TODO - speedup this with multiprocessing
        for img in detImgs :
            attr_img = []
            print(img)
            det_i = dets.loc[dets['imageName'] == img]
            for i, row in det_i.iterrows():
                r = row.to_dict()
                try :
                    attr_i = attrs["imgs"].index(r["timgName"])
                    attr_img.append({
                        "bbox" : r["bbox"],
                        "attr" : self.processAttrs(attrs["attrs"][attr_i])
                    })
                except Exception as e:
                    print(F"unable to extract data for {r['timgName']}, failed with exception {e}")
                    # raise
            # save the attr_img file in metadata dir
            fName = os.path.join(imgMetadataDir,F"{img.split('.')[0]}.json")
            # print(fName)
            with open(fName,'w') as fd:
                json.dump(attr_img, fd)

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
    parser.add_argument("--batch_size",type=int, help="batch size used for traning", required=False,default=100)


    args = parser.parse_args()

    mg = MetadataGenerator(args)
    mg.runByteTracker()
    mg.generateMetadaForEachImg()