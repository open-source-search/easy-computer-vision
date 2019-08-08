THIS PLATFORM IS IN DEVELOPMENT. THIS MESSAGE WILL BE REMOVED AS SOON AS STABLE WORKING VERSION WILL BE RELEASED.

# OSS Easy Computer Vision
OSS Easy Computer Vision is an open source platform (part of OpenSourceSearch.com University's "Easy" series),
built around Mask R-CNN algorithm. It is designed for those who would like to be able to train their own machine
learning models for object detection on images or video, but do not know or want to code their own training/prediction
pipeline. OSS Easy Computer Vision hides all complexity under it's hood and provides a very detailed documentation,
making it as easy as possible for everyone to get involved training their models for computer vision. It supports
object detection on image, directory of images, video file and live video stream.

# Installation Instructions
## Install on Ubuntu Linux
TBD

## Install on Mac OS
TBD

# Running Predictions

## Prerequisite Steps
Before you could run a prediction against an image or a video, you first need to specify a path to the directory with
all files belonging to the model, which you chose to be the golden egg model.

Path to the model has to be specified in the "prediction.json" config file under key: "MODEL_DIR".
"prediction.json" config is located at the following path:
```path/to/easy-computer-vision/configs/prediction.json```

## Predict Against an Image File
To predict against an image please run following steps:
1) cd path/to/easy-computer-vision/
2) python3 -m predict --image /path/to/image/file.jpg

## Predict Against a Video File
To predict against a video please run following steps:
1) cd path/to/easy-computer-vision/
2) python3 -m predict --video /path/to/video/file.mp4

## Predict Against a Video Stream
To predict against a live video stream run following steps:
1) cd path/to/easy-computer-vision/
2) python3 -m predict --video http://www.somedomain.com/video/stream/

## Predict Against a Video From Webcam
To predict against a video feed coming from webcam please run following steps:
1) cd path/to/easy-computer-vision/
2) python3 -m predict --video 0

## Where do the output files get saved?
Output JSON file with predictions and output image/video file with predictions rendered on it are saved into:
```path/to/easy-computer-vision/output/```
Unique filename is generated automatically using a timestamp.
We save prediction into an output image or video file by default. However you could switch it off by specifying the
following flag in the prediction command:
```--nooutput```
Full command will then look this way:
```python3 -m predict --nooutput --image /path/to/image/file.jpg```

# References
We forked the following implementation of Mask R-CNN algorithm and built framework around it:
```https://github.com/matterport/Mask_RCNN```

In OSS Easy Computer Vision, the contents of "mask_r_cnn" directory are as they were at the time when we cloned it on
July 31st, 2019 from:
```https://github.com/matterport/Mask_RCNN```

We included the original "Mask_RCNN" code in our repository to prevent changes to the original from breaking our code.
We have done slight changes to the original "Mask_RCNN" implementation.