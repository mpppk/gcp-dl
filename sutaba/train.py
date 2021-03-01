import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from common.vgg16 import fit


def check_images(image_dir_path: str, generator):
    for v in tqdm(generator.filenames):
        try:
            Image.open(image_dir_path + '/' + v)
        except Exception as e:
            print(e)


def train(
        image_dir_path: str,
        model_path: str,
        img_width: int,
        img_height: int,
        batch_size: int,
        validation_split: float = 0.25,
):
    classes = ["sutaba", "ramen", "other"]
    datagen = ImageDataGenerator(
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1.0 / 255.0,
        validation_split=validation_split
     )

    train_generator = datagen.flow_from_directory(
        directory=image_dir_path,
        subset="training",
        batch_size=batch_size,
        classes=classes,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(img_width, img_height),
    )


    validation_generator = datagen.flow_from_directory(
        directory=image_dir_path,
        subset="validation",
        batch_size=batch_size,
        classes=classes,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(img_width, img_height),
    )

    # check_images(image_dir_path, train_generator)
    # check_images(image_dir_path, validation_generator)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    fit(
        train_generator,
        validation_generator,
        img_width,
        img_height,
        {'sutaba': 973, 'ramen': 617, 'other': 2474},
        model_path,
        batch_size,
        1000,
        channel=3
    )


if __name__ == '__main__':
    base_dir = '/mnt/disks/sutaba/'
    train(
        # image_dir_path='/Users/yuki/ghq/github.com/mpppk/twitter/old_images',
        # image_dir_path='/mnt/disks/sutaba/old_images',
        image_dir_path=base_dir+'old/images',
        # image_dir_path='/home/yuki/dataset/sutaba/old_images',
        model_path=base_dir+'old/models',
        img_width=225,
        img_height=225,
        batch_size=512,
        validation_split=0.25
    )
