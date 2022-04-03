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
        boxes = np.array(data["boxes"][i_d])[labelIdxs].tolist()
        scores = np.array(data["scores"][i_d])[labelIdxs].tolist()

        dets = []
        for i,b in enumerate(boxes) :
            b.append(scores[i])
            dets.append(b)


        outDetections["personDetections"].append({
            "img_name" : img,
            "detections" : dets
        })
    
    # For BYTETRACK
    # difference between img_info and img_size ref (https://github.com/ifzhang/ByteTrack/issues/23)
    # img_info is the origin size and img_size is the inference size.

    print(len(outDetections["personDetections"]))
    # print(personDets)

    return outDetections













if __name__ == "__main__":
    pklFile = "/home/akunchala/Documents/PhDStuff/privacyFrameworkWithMetadata/tmp/objectDetection.pkl"
    getPersonsFromPkl(pklFile)