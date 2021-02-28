from tqdm import tqdm
from PIL import Image
import glob
import os
import json


def check_images(glob_path: str):
    for file in tqdm(glob.glob(glob_path)):
        try:
            Image.open(file)
        except Exception as e:
            print(json.dumps({'path': os.path.basename(file), 'boundingBoxes': [{'tagName': 'invalid'}]}))


if __name__ == '__main__':
    check_images('/Users/yuki/ghq/github.com/mpppk/twitter/images/*.jpg')
