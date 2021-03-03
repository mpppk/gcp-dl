import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np

from common.vgg16 import fit

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(
        train_df,
        image_dir_path: str,
        model_path: str,
        img_width: int,
        img_height: int,
        batch_size: int,
        validation_split: float = 0.25,
):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=validation_split)
    counts = train_df["label"].value_counts(normalize=True)

    x_col, y_col = "path", "label"
    train_df = train_df.sample(frac=1, random_state=42)
    print(train_df.head(2))

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir_path,
        x_col=x_col,
        y_col=y_col,
        subset="training",
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(img_width, img_height),
    )

    validation_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir_path,
        x_col=x_col,
        y_col=y_col,
        subset="validation",
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(img_width, img_height),
    )

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    fit(
        train_generator,
        validation_generator,
        img_width,
        img_height,
        counts,
        model_path,
        batch_size,
        100,
        channel=3
    )


if __name__ == '__main__':
    csv_path = 'assets/20210303assets.csv'
    df = pd.read_csv(csv_path)
    df = df[~df["tags"].str.contains('invalid', na=False)]
    df = df[~df["tags"].str.contains('predict', na=False)]
    df = df.dropna(how='any')

    def select_label(tags: str):
        labels = [
            'sutaba:bev',
            'sutaba:frap',
            'sutaba:food',
            'sutaba:other',
            'other',
            'food',
            'ramen',
            'men',
            'sushi',
            'niku',
            'cafe',
            'sake',
        ]
        for label in labels:
            if label in tags:
                return label
        return np.nan

    df = df.assign(
        label=df['tags'].apply(select_label)
    )
    df = df.dropna(how='any')
    print(df.tail())
    print(len(df))

    # base_dir = '/mnt/disks/sutaba/'
    base_dir = './'
    train(
        train_df=df,
        image_dir_path='/Users/yuki/ghq/github.com/mpppk/twitter/images',
        model_path=base_dir+'models',
        img_width=225,
        img_height=225,
        batch_size=512,
        validation_split=0.25
    )
