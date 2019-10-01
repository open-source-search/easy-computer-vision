# OSS Easy Computer Vision
OSS Easy Computer Vision is an open source platform (part of OpenSourceSearch.com University's "Easy" series), built around Mask R-CNN algorithm. It is designed for those who would like to be able to train their own machine learning models for object detection on images or video, but do not know how or want to code their own training/prediction pipelines. OSS Easy Computer Vision hides all complexity under it's hood and provides a very detailed documentation, making it as easy as possible for everyone to get involved training their models for computer vision. It supports object detection on image, directory of images, video file and live video stream.


# Installation Instructions
## Install on Ubuntu Linux
### Install Python3
Ubuntu Linux versions 17.10 and 18.04 come wit Python3 preinstalled.
Ubuntu Linux versions 17.04 and below, do not come wit Python3 preinstalled.
If you are on Ubuntu 17.04 or lower, first install Python3.
Run following commands in the Konsole:
1) sudo apt-get update
2) sudo apt-get install python3.6
3) Test your installation by opening CMD and running command: "python3".

### Check out OSS Easy Computer Vision code to your Ubuntu Linux machine:
1) Open command prompt.
2) cd to directory where you want code to be located on your Ubuntu Linux machine.
3) Run command: "git clone https://github.com/open-source-search/easy-computer-vision.git"

### Install required modules/libraries:
Run following commands in the Konsole:
1) cd path/to/easy-computer-vision/
2) sudo python3 -m pip install -r requirements.txt


## Install on Mac OS
### Install Python3
Run following commands in the Terminal:
1) xcode-select --install
2) /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
3) brew install python3
4) Test your installation by opening CMD and running command: "python3".

### Check out OSS Easy Computer Vision code to your Mac computer:
1) Open command prompt.
2) cd to directory where you want code to be located on your windows computer.
3) Run command: "git clone https://github.com/open-source-search/easy-computer-vision.git"

### Install required modules/libraries:
Run following commands in the Terminal:
1) cd path/to/easy-computer-vision/
2) sudo python3 -m pip install -r requirements.txt


## Install on Windows
### Install Python3 for Windows:
1) Download Python3 executable installer from: "https://www.python.org/downloads/windows/".
2) Double click on installer .exe file and follow the instructions to install Python3 on your Windows.
3) Test your installation by opening CMD and running command: "py".

### Check out OSS Easy Computer Vision code to your Windows computer:
1) Open command prompt.
2) cd to directory where you want code to be located on your windows computer.
3) Run command: "git clone https://github.com/open-source-search/easy-computer-vision.git"

### Install required modules/libraries:
1) cd C:\path\to\easy-computer-vision\
2) py -m utils.get-pip
3) py -m pip install --upgrade pip
4) py -m pip install -r requirements.txt

NOTE: On Windows, please use following format for the commands:
1) py -m train.start_training
2) py -m predict.run_prediction --image C:\path\to\image\file.jpg
... etc. So from below instructions, instead of typing a command as "python3 -m train", on Windows,
type it as "py -m train". Everything else stays the same.


# Preparing Images for Training (Labeling Data)
We suggest to use the labeling tool called RectLabel to label your data. RectLabel could be downloaded on the website:
"https://rectlabel.com". It's an inexpensive tool, which lets you label with polygons and lets you export your labels
in COCO format, which generates the "annotations.json" file that this framework requires for training.

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
To start a training of your model, please execute following 2 commands in the same order:
1) cd path/to/easy-computer-vision/
2) python3 -m train.start_training


# Choosing the best performing model

Training pipeline will generate a model for each epoch. So for instance if you train for 100 epochs, you will have 100
model dumps stored in the path: ```path/to/easy-computer-vision/models/```. You will need to choose only one model that
is expected to perform best. To help you with that choice, we prepared a log parser script that will parse through logs,
created while model was training and will output the following into the console:
```
Logs filename:  my_logs_file.log
Lowest val_mrcnn_mask_loss is 0.0469 at epoch 210/450.
Lowest val_mrcnn_class_loss is 0.0081 at epoch 215/450.
Lowest val_rpn_class_loss is 0.001 at epoch 12/450.
Lowest mrcnn_bbox_loss is 0.1013 at epoch 215/450.
Lowest val_rpn_bbox_loss is 0.0204 at epoch 254/450.
Lowest loss is 0.7116 at epoch 215/450.
Lowest val_loss is 0.1129 at epoch 254/450.
Lowest rpn_bbox_loss is 0.1884 at epoch 255/450.
Lowest rpn_class_loss is 0.0054 at epoch 218/450.
Lowest mrcnn_class_loss is 0.1414 at epoch 215/450.
Lowest mrcnn_mask_loss is 0.2525 at epoch 216/450.
Lowest val_mrcnn_bbox_loss is 0.018 at epoch 248/450.
```

The parameter: "Lowest val_loss is at epoch 254/450." points to the model dump generated at 254th epoch, which would be
your best performing model if you don't want to run additional tests. However, the best way to do it, would be, to take
all models that script outputs and run predictions against all of them to test each model dump, running it through
actual predictions. Model, which would produce best actual prediction accuracies, would be your best performing model.
You would need only one model, therefore the rest of the ".h5" model files generated by the training pipeline could be
deleted.

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


# Running Predictions In Terminal/Konsole/CMD

## Prerequisite Steps
Before you could run a prediction against an image or a video, you first need to specify a paths to the model itself
and the files that are required for the model to be loaded and predicted against.

Paths to required files should be specified in the prediction config file.
"prediction.json" config is located at the following path:
```path/to/easy-computer-vision/configs/prediction.json```

There are 3 paths that should be specified.
1) "MODEL_FILE_PATH" - Is the absolute path to the model file with file extension: ".h5"
2) "ANNOTATIONS_FILE_PATH" - Is the path to the COCO "annotations.json" file that was exported by RectLabel tool, which
we suggested to be used to label training dataset in section: "Preparing Images for Training (Labeling Data)" of this
README file.
3) "PARAMS_FILE_PATH" - Is the path to the "training.json" config file, located at the path: "configs/training.json" of
this platform. Its basically is the config file you used to train the model.

NOTE: It is suggested to create a separate directory, for example: "path/to/platform/model/date_stamp/", and copy above
3 files into that directory. It is important to preserve the "training.json" and "annotations.json" in the same state
as was used to train the model. If you change them, model wo't load. Therefore the rule of thumb is to copy the golden
egg model and 2 of the required configs into separate dated directory. So you have all configs that wee used to train
the model backed up and stores separately in the same directory with the model. Later, if you modify original configs
and retrain, files for previously trained models will stay unchaged.

## Predict Against an Image File
To predict against an image please run following steps:
1) cd path/to/easy-computer-vision/
2) python3 -m predict.run_prediction --image /path/to/image/file.jpg

## Predict Against a Video File
To predict against a video please run following steps:
1) cd path/to/easy-computer-vision/
2) python3 -m predict.run_prediction --video /path/to/video/file.mp4

## Predict Against a Video Stream
To predict against a live video stream run following steps:
1) cd path/to/easy-computer-vision/
2) python3 -m predict.run_prediction --video http://www.somedomain.com/video/stream/

## Predict Against a Video From Webcam
To predict against a video feed coming from webcam please run following steps:
1) cd path/to/easy-computer-vision/
2) python3 -m predict.run_prediction --video 0

## Where do prediction output files get saved?
Output JSON file with predictions and output image/video file with predictions rendered on it are saved into:
```path/to/easy-computer-vision/output/```
Unique filename is generated automatically using a timestamp.
We save prediction into an output image or video file by default. However you could switch it off by specifying the
following flag in the prediction command:
```--nooutput```
Full command will then look this way:
```python3 -m predict.run_prediction --nooutput --image /path/to/image/file.jpg```


# Running Predictions using RESTful API

As an example of how can predictions be made over the RESTful API, we have prepared following example that runs on Flask.
We also prepared a script that sends an image file to API for upload, then outputs prediction results into console.
When file is uploaded to API it is uploaded to the path: ```path/to/easy-computer-vision/uploads/```. However after
prediction has been made, uploaded file gets deleted.

## Start API:
To start API, please run following commands:
1) cd path/to/easy-computer-vision/
2) python3 -m api.api

## To run a test script, which passes image file to API for upload and prediction, run following commands:
1) cd path/to/easy-computer-vision/
2) python3 -m utils.post_file --image /path/to/image/file.jpg

## Where do prediction output files get saved?
Output JSON file with predictions and output image/video file with predictions rendered on it are saved into:
```path/to/easy-computer-vision/output/```
Unique filename is generated automatically using a timestamp.
We save prediction into an output image or video file by default. However you could switch it off by specifying the
following flag in the prediction command:
```--nooutput```
Full command will then look this way:
```python3 -m predict.run_prediction --nooutput --image /path/to/image/file.jpg```

NOTE: Each time you call for predictions, model is loading from scratch, therefore performance is not as fast as it
could be if you would have loaded the model into memory and had it ready as a static object to call for predictions.
We will add this feature in one of our next builds. For now as is.


# References
We forked the following implementation of Mask R-CNN algorithm and built framework around it:
```https://github.com/matterport/Mask_RCNN```

In OSS Easy Computer Vision, the contents of "mask_r_cnn" directory are as they were at the time when we cloned it on
July 31st, 2019 from:
```https://github.com/matterport/Mask_RCNN```

We included the original "Mask_RCNN" code in our repository to prevent changes to the original from breaking our code.
We have done slight changes to the original "Mask_RCNN" implementation.