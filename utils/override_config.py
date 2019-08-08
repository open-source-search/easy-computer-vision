

import json

from mask_r_cnn.mrcnn.config import Config
from utils.global_storage import GlobalStorage


class OverrideConfig(Config):

    JSON_CONFIG_PATH = GlobalStorage.network_params

    with open(JSON_CONFIG_PATH, "r") as file:
        params = json.loads(file.read())

    for param in params:
        vars()[param] = params[param]