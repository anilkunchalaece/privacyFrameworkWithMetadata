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

## Notes
- Do we need to look for SOTA pedestrian attribute recognision?
  - I'm currently using transfer learning to make a model. May be for later stage, we can look into [Rethinking of PAR](https://github.com/valencebond/Rethinking_of_PAR)

