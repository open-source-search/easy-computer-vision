import os
import json
import numpy as np
from PIL import Image, ImageDraw
import skimage

import mask_r_cnn.mrcnn.utils as utils


class LoadDataset(utils.Dataset):

    SOURCE_NAME = "OSS-ECV"
    CATEGORIES = "categories"
    ID = "id"
    NAME = "name"
    ANNOTATIONS = "annotations"
    IMAGE_ID = "image_id"
    IMAGES = "images"
    FILE_NAME = "file_name"
    WIDTH = "width"
    HEIGHT = "height"
    CATEGORY_ID = "category_id"
    SEGMENTATION = "segmentation"

    def load_data(self, annotation_json_path=None, images_dir=None):

        assert annotation_json_path and images_dir, "Path to annotations.json and path to " \
                                               "images directory must be both provided."

        with open(annotation_json_path, "r") as file:
            annotation_json = json.loads(file.read())

        self.add_classes(annotation_json)
        self.add_images(annotation_json, images_dir)

    def add_classes(self, annotation_json):
        categories = annotation_json[self.CATEGORIES]
        for category in categories:
            id = category[self.ID]
            name = category[self.NAME]
            if id > 0:
                self.add_class(self.SOURCE_NAME, id, name)

    def add_images(self, annotation_json, images_dir):
        annotations = self.__get_annotations_with_image_id(annotation_json)
        image_storage = {}
        for image in annotation_json[self.IMAGES]:
            image_id = image[self.ID]
            if not image_id in image_storage:
                image_storage[image_id] = image
                try:
                    image_file_name = image[self.FILE_NAME]
                    image_width = image[self.WIDTH]
                    image_height = image[self.HEIGHT]
                except:
                    image_file_name = None
                    image_width = None
                    image_height = None

                image_path = os.path.join(images_dir, image_file_name)
                image_annotations = annotations[image_id]

                self.add_image(
                    source=self.SOURCE_NAME,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def __get_annotations_with_image_id(self, annotation_json):
        annotations = annotation_json[self.ANNOTATIONS]
        anntn_storage = {}
        for annotation in annotations:
            image_id = annotation[self.IMAGE_ID]
            try:
                anntn_list = anntn_storage[image_id]
                anntn_list.append(annotation)
                anntn_storage[image_id] = anntn_list
            except:
                anntn_storage[image_id] = [annotation]
        return anntn_storage

    def load_mask(self, image_id):

        image_info = self.image_info[image_id]
        annotations = image_info[self.ANNOTATIONS]

        masks_storage = []
        class_ids_storage = []
        for annotation in annotations:
            class_id = annotation[self.CATEGORY_ID]
            image_width = image_info[self.WIDTH]
            image_height = image_info[self.HEIGHT]
            image = Image.new(mode='1',
                              size=(image_width, image_height))
            image_draw = ImageDraw.ImageDraw(image)

            for segmentation in annotation[self.SEGMENTATION]:
                image_draw.polygon(segmentation, fill=1)
                masks_storage.append(np.array(image) > 0)
                class_ids_storage.append(class_id)

        return np.dstack(masks_storage), \
               np.array(class_ids_storage, dtype=np.int32)