import skimage
import cv2
import datetime
import os
import json
import argparse
import numpy
import logging
logging.getLogger().setLevel(logging.INFO)

from mask_r_cnn.mrcnn import model as modellib
from mask_r_cnn.mrcnn import visualize

from utils.global_storage import GlobalStorage
from utils.parse_logs import ParseLogs


DEFAULT_LOGS_DIR = "logs/"


class Predict():

    CONFIG = "configs/prediction.json"
    MODEL_FILE_PATH = "MODEL_FILE_PATH"
    ANNOTATIONS_FILE_PATH = "ANNOTATIONS_FILE_PATH"
    PARAMS_FILE_PATH = "PARAMS_FILE_PATH"
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
    BOUNDING_BOX = 'bounding_box'
    LABEL = 'label'
    ACCURACY = 'accuracy'
    PREDICTIONS = 'predictions'

    def __init__(self):

        with open(self.CONFIG, "r") as file:
            self.prediction_config = json.loads(file.read())

        self.weights_path = self.prediction_config[self.MODEL_FILE_PATH]
        self.annotations_path = self.prediction_config[self.ANNOTATIONS_FILE_PATH]
        GlobalStorage.network_params = self.prediction_config[self.PARAMS_FILE_PATH]

        from utils.override_config import OverrideConfig
        self.config = OverrideConfig()

        logs_file = "predict_{:%Y%m%dT%H%M%S}.log".format(datetime.datetime.now())
        if not os.path.exists(self.config.LOGS_DIR):
            os.makedirs(self.config.LOGS_DIR)
        ParseLogs.redirect_logs_to_file(os.path.join(self.config.LOGS_DIR, logs_file))

        if not os.path.exists(self.OUTPUT_PATH):
            os.makedirs(self.OUTPUT_PATH)


    def load_model(self):
        model = modellib.MaskRCNN(mode=self.MODE, config=self.config,
                                  model_dir=os.path.dirname(self.weights_path))
        model.load_weights(self.weights_path, by_name=True)
        return model

    def __get_cass_names(self):
        with open(self.annotations_path, "r") as file:
            data = json.loads(file.read())
        return {category[self.ID]:category[self.NAME] for category in data[self.CATEGORIES]}

    def __fix_image(self, image_obj):
        image_narray = numpy.array(image_obj)
        if image_narray.shape[-1] == 4:
            image_narray = image_narray[..., :3]
        return image_narray

    def __reformat_metadata(self, boxes, class_ids, class_names, scores):
        output = []
        for idx, id in enumerate(class_ids):
            output.append({self.BOUNDING_BOX: [int(value) for value in boxes[idx]],
                           self.ACCURACY: float(scores[idx]),
                           self.LABEL: class_names[class_ids[idx]]})
        return output

    def __save_metadata(self, metadata, filepath):
        data = json.dumps(metadata)
        with open(filepath, "w") as file:
            file.write(data)

    def predict(self, model, image_path=None, video_path=None, nooutput=None):
        assert image_path or video_path

        if image_path:
            image = skimage.io.imread(image_path)
            results = model.detect([self.__fix_image(image)], verbose=1)[0]
            file_name_stamp = "image_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())

            boxes = results[self.ROIS]
            masks = results[self.MASKS]
            class_ids = results[self.CLASS_IDS]
            class_names = self.__get_cass_names()
            scores = results[self.SCORES]

            metadata = {self.PREDICTIONS: self.__reformat_metadata(boxes, class_ids, class_names, scores)}

            if not nooutput:
                splash = visualize.display_instances(image, boxes, masks, class_ids, class_names, scores)
                skimage.io.imsave(os.path.join(self.OUTPUT_PATH, file_name_stamp+".png"), splash)
                self.__save_metadata(metadata, os.path.join(self.OUTPUT_PATH, file_name_stamp+".json"))


            return metadata

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
                    results = model.detect([self.__fix_image(image)], verbose=0)[0]

                    boxes = results[self.ROIS]
                    masks = results[self.MASKS]
                    class_ids = results[self.CLASS_IDS]
                    class_names = self.__get_cass_names()
                    scores = results[self.SCORES]

                    splash = visualize.display_instances(image, boxes, masks, class_ids, class_names, scores)

                    vwriter.write(splash)
                    count += 1
            vwriter.release()

            return None



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
    pr.predict(model, image_path=args.image, video_path=args.video, nooutput=args.nooutput)
