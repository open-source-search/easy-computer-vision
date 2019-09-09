import numpy as np
import imgaug.augmenters as iaa
import json

CONFIG = "configs/training.json"


class ImageAugmentation():

    AUG_PERCENTAGE = 0.5
    AUGMENT_DATA = "AUGMENT_DATA"

    @staticmethod
    def augment():
        with open(CONFIG, "r") as file:
            config = json.loads(file.read())
        if config[ImageAugmentation.AUGMENT_DATA]:
            matrix = np.array([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]])
            return iaa.Sometimes(ImageAugmentation.AUG_PERCENTAGE, [
                                 iaa.GaussianBlur(sigma=2.0),
                                 iaa.Sequential([iaa.Affine(rotate=45), iaa.Sharpen(alpha=1.0)]),
                                 iaa.WithColorspace(
                                    to_colorspace="HSV",
                                    from_colorspace="RGB",
                                    children=iaa.WithChannels(0, iaa.Add((10, 50)))
                                 ),
                                 iaa.AdditiveGaussianNoise(scale=0.2 * 255),
                                 iaa.Add(50, per_channel=True),
                                 iaa.Sharpen(alpha=0.5),
                                 iaa.WithChannels(0, iaa.Add((10, 100))),
                                 iaa.WithChannels(0, iaa.Affine(rotate=(0, 45))),
                                 iaa.Noop(),
                                 iaa.Superpixels(p_replace=0.5, n_segments=64),
                                 iaa.Superpixels(p_replace=(0.1, 1.0), n_segments=(16, 128)),
                                 iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                                 iaa.WithChannels(0, iaa.Add((50, 100))),
                                 iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
                                 iaa.Grayscale(alpha=(0.0, 1.0)),
                                 iaa.GaussianBlur(sigma=(0.0, 3.0)),
                                 iaa.AverageBlur(k=(2, 11)),
                                 iaa.AverageBlur(k=((5, 11), (1, 3))),
                                 iaa.MedianBlur(k=(3, 11)),
                                 iaa.Convolve(matrix=matrix),
                                 iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
                                 iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
                                 iaa.EdgeDetect(alpha=(0.0, 1.0)),
                                 iaa.DirectedEdgeDetect(alpha=(0.0, 1.0), direction=(0.0, 1.0)),
                                 iaa.Add((-40, 40)),
                                 iaa.Add((-40, 40), per_channel=0.5),
                                 iaa.AddElementwise((-40, 40)),
                                 iaa.AddElementwise((-40, 40), per_channel=0.5),
                                 iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
                                 iaa.AdditiveGaussianNoise(scale=0.05 * 255),
                                 iaa.AdditiveGaussianNoise(scale=0.05 * 255, per_channel=0.5),
                                 iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                 iaa.Dropout(p=(0, 0.2)),
                                 iaa.Dropout(p=(0, 0.2), per_channel=0.5),
                                 iaa.CoarseDropout(0.02, size_percent=0.5),
                                 iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
                                 iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
                                 iaa.Invert(0.25, per_channel=0.5),
                                 iaa.Invert(0.5),
                                 iaa.ContrastNormalization((0.5, 1.5)),
                                 iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
                                 iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)
                                 ])
        else:
            return None
