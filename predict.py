import skimage
import cv2
import datetime
import os
import json
import argparse

from mask_r_cnn.mrcnn import model as modellib
from mask_r_cnn.mrcnn import visualize

from utils.override_config import OverrideConfig
from utils.global_storage import GlobalStorage


DEFAULT_LOGS_DIR = "logs/"


class Predict():

    CONFIG = "configs/prediction.json"
    MODEL_DIR = "MODEL_DIR"
    MODEL_FILENAME = "master.h5"
    ANNOTATIONS_FILENAME = "annotations.json"
    PARAMS_FILENAME = "params.json"
    MODE = "inference"
    OUTPUT_PATH = "output/"
    ROIS = "rois"
    MASKS = "masks"
    CLASS_IDS = "class_ids"
    SCORES = "scores"
    ID = "id"
    NAME = "name"
    CATEGORY = "category"
    CATEGORIES = "categories"

    def __init__(self):
        with open(self.CONFIG, "r") as file:
            self.model_dir_path = json.loads(file.read())[self.MODEL_DIR]
        self.weights_path = os.path.join(self.model_dir_path, self.MODEL_FILENAME)
        self.annotations_path = os.path.join(self.model_dir_path, self.ANNOTATIONS_FILENAME)
        GlobalStorage.network_params = os.path.join(self.model_dir_path, self.PARAMS_FILENAME)
        self.config = OverrideConfig()
        if not os.path.exists(self.OUTPUT_PATH):
            os.makedirs(self.OUTPUT_PATH)

    def load_model(self):
        model = modellib.MaskRCNN(mode=self.MODE, config=self.config,
                                  model_dir=self.model_dir_path)
        model.load_weights(self.weights_path, by_name=True)
        return model

    def __get_cass_names(self):
        with open(self.annotations_path, "r") as file:
            data = json.loads(file.read())
        return {category[self.ID]:category[self.NAME] for category in data[self.CATEGORIES]}

    def predict(self, model, image_path=None, video_path=None):
        assert image_path or video_path

        if image_path:
            image = skimage.io.imread(image_path)

            results = model.detect([image], verbose=1)[0]
            file_name = "image_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())

            boxes = results[self.ROIS]
            masks = results[self.MASKS]
            class_ids = results[self.CLASS_IDS]
            class_names = self.__get_cass_names()
            scores = results[self.SCORES]

            splash = visualize.display_instances(image, boxes, masks, class_ids, class_names, scores)

            skimage.io.imsave(os.path.join(self.OUTPUT_PATH, file_name), splash)
            print()

        elif video_path:
            vcapture = cv2.VideoCapture(video_path)
            width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = vcapture.get(cv2.CAP_PROP_FPS)

            file_name = "video_{:%Y%m%dT%H%M%S}.mp4".format(datetime.datetime.now())
            vwriter = cv2.VideoWriter(os.path.join(self.OUTPUT_PATH, file_name),
                                      cv2.VideoWriter_fourcc(*'MP4V'),
                                      fps, (width, height), True)

            count = 0
            success = True
            while success:
                success, image = vcapture.read()
                if success:
                    image = image[..., ::-1]
                    results = model.detect([image], verbose=0)[0]

                    boxes = results[self.ROIS]
                    masks = results[self.MASKS]
                    class_ids = results[self.CLASS_IDS]
                    class_names = self.__get_cass_names()
                    scores = results[self.SCORES]

                    splash = visualize.display_instances(image, boxes, masks, class_ids, class_names, scores)

                    vwriter.write(splash)
                    count += 1
            vwriter.release()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prediction engine for Easy Computer Vision Framework')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='path or URL to image')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video, for webcam specify --video 0",
                        help='path or URL to video, for webcam specify --video 0')
    parser.add_argument('--nooutput', default=False,
                        action="store_true",
                        help='specify this flag to switch off the output file generation')
    args = parser.parse_args()

    assert args.image or args.video, \
        "Provide --image or --video for prediction"

    pr = Predict()
    model = pr.load_model()
    pr.predict(model, image_path=args.image, video_path=args.video)
