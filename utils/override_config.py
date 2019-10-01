import json

from mask_r_cnn.mrcnn.config import Config
from utils.global_storage import GlobalStorage


class OverrideConfig(Config):

    try:
        JSON_CONFIG_PATH = GlobalStorage.network_params

        with open(JSON_CONFIG_PATH, "r") as file:
            params = json.loads(file.read())

        for param in params:
            vars()[param] = params[param]

        with open(params["TRAINING_ANNOTATIONS_FILE"], "r") as file:
            NUM_CLASSES = len(json.loads(file.read())["categories"]) + 1
            print("NUM_CLASSES: ", NUM_CLASSES)

        MINI_MASK_SHAPE = tuple(params["MINI_MASK_SHAPE"])
    except:
        pass
