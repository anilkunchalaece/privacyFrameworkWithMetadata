"""
This class is used as a middle layer to process tracklets and generate metadata for each image
Tracklets are saved based on tracking_id where each tracklet is defined using seperate dir with tracklet number
and each image is saving with following naming convention
        {t_id}_{imageName}.{jpg/png}
Functionalities
    1. Segregate all tracklets in a list for attrNet inference
    2. Generate bbox reference for tracklet image names and bbox
"""

class AttrGenPreprocessing:
    def __init__(self):
        pass