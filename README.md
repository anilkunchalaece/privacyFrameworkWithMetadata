# privacyFrameworkWithMetadata

## Proposed system
![alt proposed system](readMeImgs/proposedOverview.png)

## Progress 
- Integrated [ByteTrack](https://github.com/ifzhang/ByteTrack) for tracking pedestrians across the frames
- Using Faster-rcnn for object detection and seperating the all the pedestrians from it


## TODO
- Replace SORT with ByteTrack in VIBE
- AttrNet
  - Following from [Rethinking of PAR](https://github.com/valencebond/Rethinking_of_PAR) and [Strong Baseline of Pedestrian Attribute Recognistion](https://github.com/aajinjin/Strong_Baseline_of_Pedestrian_Attribute_Recognition) , Input pedestrian image size is considered as 256(height) x 192(width)
- ### Classifier 
  - Following are the few designs for the classifiers
    - [pedestrian-attribute-recognision-pytorch](https://github.com/dangweili/pedestrian-attribute-recognition-pytorch)
      - A Single FC layer with i/p 2048 and o/p of no.of pedestrian attributes [ref](https://github.com/dangweili/pedestrian-attribute-recognition-pytorch/blob/468ae58cf49d09931788f378e4b3d4cc2f171c22/baseline/model/DeepMAR.py#L41)
    - [Strong_Baseline_of_Pedestrian_Attribute_Recognition](https://github.com/aajinjin/Strong_Baseline_of_Pedestrian_Attribute_Recognition)
      - A single FC layer with i/9 2048 and o/p of no.of pedestrian attributes [ref](https://github.com/aajinjin/Strong_Baseline_of_Pedestrian_Attribute_Recognition/blob/4b1afcc76b4bbc116f6648f4fd9fbe18502390ee/models/base_block.py#L11)

## Notes
- Do we need to look for SOTA pedestrian attribute recognision?
  - I'm currently using transfer learning to make a model. May be for later stage, we can look into [Rethinking of PAR](https://github.com/valencebond/Rethinking_of_PAR)

- Tracklets bbox image naming convention
  - F"t_{tIds[i]}-{os.path.basename(imgName)}"