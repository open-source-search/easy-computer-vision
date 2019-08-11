import os
import urllib.request


class DownloadCocoDataset():

    DOWNLOAD_DIR = "coco_weights/"
    DOWNLOAD_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"

    def __init__(self):
        if not os.path.exists(self.DOWNLOAD_DIR):
            os.makedirs(self.DOWNLOAD_DIR)

    def download(self):
        urllib.request.urlretrieve(self.DOWNLOAD_URL, os.path.join(self.DOWNLOAD_DIR, "mask_rcnn_coco.h5"))


if __name__ == '__main__':
    dcd = DownloadCocoDataset()
    dcd.download()