from mask_r_cnn.mrcnn import model as modellib, utils

from utils.override_config import OverrideConfig
from utils.global_storage import GlobalStorage
from utils.load_dataset import LoadDataset


class Train():

    CONFIG = "configs/training.json"
    COCO_WEIGHTS = "coco_weights/mask_rcnn_coco.h5"
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
    MODE = "training"

    def __init__(self):
        self.config = OverrideConfig()

        self.model = modellib.MaskRCNN(mode=self.MODE,
                                       config=self.config,
                                       model_dir=self.config.TRAINED_MODELS_DIR)

        # utils.download_trained_weights(weights_path)
        self.model.load_weights(self.COCO_WEIGHTS, by_name=True, exclude=[
                                "mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

    def train(self):

        training_dataset = LoadDataset()
        training_dataset.load_data(self.config.TRAINING_ANNOTATIONS_FILE,
                                   self.config.TRAINING_IMAGES_DIR)
        training_dataset.prepare()

        test_dataset = LoadDataset()
        test_dataset.load_data(self.config.VALIDATION_ANNOTATIONS_FILE,
                               self.config.VALIDATION_IMAGES_DIR)
        test_dataset.prepare()

        self.model.train(training_dataset,
                         test_dataset,
                         learning_rate=self.config.LEARNING_RATE,
                         epochs=self.config.TOTAL_NUMBER_OF_EPOCHS,
                         layers=self.config.LAYERS)


if __name__ == '__main__':
    tr = Train()
    tr.train()
