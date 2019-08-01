# OSS Easy Computer Vision
OSS Easy Computer Vision is an open source platform (part of OpenSourceSearch.com University's "Easy" series),
built around Mask R-CNN algorithm. It is designed for those who would like to be able to train their own machine
learning models for object detection on images or video, but do not know or want to code their own training/prediction
pipeline. OSS Easy Computer Vision hides all complexity under it's hood and provides a very detailed documentation,
making it as easy as possible for everyone to get involved training their models for computer vision. It supports
object detection on image, directory of images, video file and live video stream.

# References
We forked the following implementation of Mask R-CNN algorithm and built framework around it:
```https://github.com/matterport/Mask_RCNN```

In OSS Easy Computer Vision, the contents of "mask_r_cnn" directory are as they were at the time when we cloned it on
July 31st, 2019 from:
```https://github.com/matterport/Mask_RCNN```

We included the original "Mask_RCNN" code in our repository to prevent changes to the original from breaking our code.
You could however check out latest Mark R-CNN code and try running it with OSS Easy Computer Vision platform.
Just complete the following steps:
1) Rename the root directory into "mask_r_cnn".
2) Replace our "mask_r_cnn" dir with yours.
3) Add empty "__init__.py" file into "mask_r_cnn/" directory.
If no changes were made to the original that could cause our code to break, then it should work.
If it doesn't work, please comment and we will make necessary changes to make out code work with latest Mask R-CNN
implementation from repository:
```https://github.com/matterport/Mask_RCNN```
