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

## Install on Windows
TBD

# Preparing Images for Training (Labeling Data)
We suggest to use the following tool to label your data. It's an inexpensive tool, which lets you label with polygons
and at lets you export your labels in COCO format, which generates the annotations.json file that this framework
requires for training.

When you done labeling your images, export your labels in COCO format. Name exported file as "annotations.json" to stay
consistent with our further instructions below. Path to the "annotations.json" for both, training and test datasets,
will need to be specified in the config file from where training pipeline reads the instruction on what to train on.
Section "Training a Model > Update Config" explains it in more detail.


# Training a Model

## Training Requires Default Weights to be Downloaded
Before you proceed to train the model, you need to download default weights. Default weights will be used as a starting
point for training your model. Running below steps will download the default weights and place them into the path
required by this platform. When download of the default weights is complete, then move onto the next step: "Update Config".
Default weights file will be saved into the following path: "path/to/easy-computer-vision/coco_weights/mask_rcnn_coco.h5"

Please execute following commands in the same order:
1) cd path/to/easy-computer-vision/
2) python3 -m utils.download_coco_weights

## Update Config
Config pretty much contains all parameters required to create a DNN network. You can however leave them as they are.
The only parameters you must update are the ones that point training pipeline to the training and validation datasets.
When labeling data, RectLabel generated "annotations.json" file, which contains all meta data with X and Y coordinates
for all labeled objects on the image. You will need to specify an absolute path to annotations.json for training and
validation datasets. You will also need to specify path to the directory of images, which you were labeling.
Config parameters that contain the paths are:
```
"TRAINING_ANNOTATIONS_FILE": "/path/to/training/dataset/annotations.json",
"TRAINING_IMAGES_DIR": "/path/to/training/dataset/images/",
"VALIDATION_ANNOTATIONS_FILE": "/path/to/validation/dataset/annotations.json",
"VALIDATION_IMAGES_DIR": "/path/to/validation/dataset/images/",
```
The rest of the parameters, if you understand what they are, you could also modify. If you do not understand what they
are yet, no worries. Just specify paths to datasets as explained above and start the training. Rest we built to be
automated for parameters, which are calculated from datasets and those that are not, are set to values that should
lead to a good result.

## Start Model Training
To start a training of you model, please execute following 2 commands i the same order:
1) cd path/to/easy-computer-vision/
2) python3 -m train


# Choosing the best performing model

Training pipeline will generate a model for each epoch. So for instance if you train for 100 epochs, you will have 100
model dumps stored in the path: ```path/to/easy-computer-vision/models/```. You will need to choose only one model that
is expected to perform best. To help you with that choice, we prepared a log parser script that will parse through logs,
created while model was training and will output the following into the console:
```
Logs filename:  my_logs_file.log
Lowest loss is at epoch 113/200.
Lowest rpn_class_loss is at epoch 113/200.
Lowest rpn_bbox_loss is at epoch 121/200.
Lowest mrcnn_class_loss is at epoch 113/200.
Lowest mrcnn_bbox_loss is at epoch 113/200.
Lowest mrcnn_mask_loss is at epoch 113/200.
Lowest val_loss is at epoch 79/200.
Lowest val_rpn_class_loss is at epoch 70/200.
Lowest val_rpn_bbox_loss is at epoch 123/200.
Lowest val_mrcnn_class_loss is at epoch 106/200.
Lowest val_mrcnn_bbox_loss is at epoch 81/200.
Lowest val_mrcnn_mask_loss is at epoch 142/200.
```

The parameter: "Lowest val_loss is at epoch 79/200." points to the model, which would be your best performing model if
you don't want to run additional tests. However, the best way to do it would be to take all models that script outputs
and run predictions against all of them them to test them with actual predictions. Model, which would produce best
accuracies, would be your best performing model. You would need only one model, rest of the ".h5" model files generated
by the training pipeline could be deleted.

## How to run the script to choose best model?
By default, script will look for log files in the following path: ```path/to/easy-computer-vision/logs/```.
However we added support for you to be able to pass your own absolute path to logs directory or to logs file.

### To run this script as default:
1) cd path/to/easy-computer-vision/
2) sudo python3 -m utils.parse_logs

### To run this script with your own path to logs directory:
1) cd path/to/easy-computer-vision/
2) sudo python3 -m utils.parse_logs --dirpath /path/to/logs/directory/

### To run this script as default:
1) cd path/to/easy-computer-vision/
2) sudo python3 -m utils.parse_logs --filepath /path/to/logs/file/my_logs.log


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