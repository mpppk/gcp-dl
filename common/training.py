import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from common.load import select_subset
from common.vgg16 import fit


def train(
        csv_path: str,
        image_dir_path: str,
        model_path: str,
        img_width: int,
        img_height: int,
        batch_size: int,
        validation_split: float = 0.25,
):
    df = pd.read_csv(csv_path)
    train_df = select_subset(df, "train")
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
