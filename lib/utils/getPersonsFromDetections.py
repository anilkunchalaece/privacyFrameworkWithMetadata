import os
import pickle
import numpy as np


def getPersonsFromPkl(pklFile):
    with open(pklFile,'rb') as fd:
        data = pickle.load(fd)

    outDetections = {
        "personDetections" : [],
        "img_shape" : data["img_shape"]
    }
    for i_d, img in enumerate(data["img_names"]) :
                
        labelIdxs = np.argwhere(np.array(data["labels"][i_d]) == 1).squeeze()
        bboxes = np.array(data["boxes"][i_d])[labelIdxs]
        scores = np.array(data["scores"][i_d])[labelIdxs].tolist()
        # if there is only one bbox in the image
        if bboxes.ndim == 1 :
            dets = bboxes.tolist()
            dets.append(scores)
            dets = np.expand_dims(dets, axis=0)
            outDetections["personDetections"].append({
                "img_name" : img,
                "detections" : np.array(dets)
            })            
        else :
            bboxes = bboxes.tolist()        
            dets = []            
            for i,b in enumerate(bboxes) :
                b.append(scores[i])
                dets.append(np.array(b))

            outDetections["personDetections"].append({
                "img_name" : img,
                "detections" : np.array(dets)
            })
    
    # For BYTETRACK
    # difference between img_info and img_size ref (https://github.com/ifzhang/ByteTrack/issues/23)
    # img_info is the origin size and img_size is the inference size.

    # print(len(outDetections["personDetections"]))
    # print(personDets)
    outDetections["personDetections"] = sorted(outDetections["personDetections"], key=lambda d:d["img_name"])
    return outDetections













if __name__ == "__main__":
    pklFile = "/home/akunchala/Documents/PhDStuff/privacyFrameworkWithMetadata/tmp/objectDetection.pkl"
    getPersonsFromPkl(pklFile)