import requests
import argparse
import os
import logging
logging.getLogger().setLevel(logging.INFO)


class PostFile():

    URL = 'http://127.0.0.1:5000/api/predict/'
    ALLOWED_EXTENSIONS = {'txt', 'png', 'jpg', 'jpeg', 'gif'}

    def post(self, image_path):
        file, extension = os.path.splitext(image_path)
        if not extension.replace('.', '').strip().lower() in self.ALLOWED_EXTENSIONS:
            error = "File type is not allowed. Allowed file " \
                    "types are: {}.".format(",".join(self.ALLOWED_EXTENSIONS))
            logging.error(error)
            return {"ERROR": error}

        file = open(image_path, 'rb')
        payload = {"file": file}
        try:
            response = requests.post(self.URL, files=payload)
            response = response.text
        except Exception as e:
            logging.error(e)
            response = None
        finally:
            file.close()
        logging.info(response)
        return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Post file to API and predict example for Easy Computer Vision Framework')
    parser.add_argument('--image', required=False,
                        metavar='path to image file',
                        help='path to image file')

    args = parser.parse_args()

    assert args.image, 'Provide --image for upload and prediction'

    pf = PostFile()
    pf.post(args.image)