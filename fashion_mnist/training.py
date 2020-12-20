import vgg16
import config
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime

c = config.Config()


def select_first_label(tags):
    for label in tags.split(","):
        if label != "train" and label != "test":
            return label


def select_subset(df, subset):
    subset_df = df[df["tags"].str.contains(subset, na=False)].assign(subset=subset)

    return subset_df.assign(label=subset_df["tags"].apply(select_first_label)).dropna()


df = pd.read_csv(c.csv_path)
train_df = select_subset(df, "train")
test_df = select_subset(df, "test")

datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.25)
counts = train_df["label"].value_counts(normalize=True)

train_df = train_df.sample(frac=1, random_state=42)


args = {
    "x_col": c.x_col,
    "y_col": c.y_col,
    "directory": c.image_dir_path,
    "batch_size": c.batch_size,
    "seed": c.seed,
    "class_mode": "categorical",
    "targeet_size": (c.img_width, c.img_height),
    "shuffle": True,
}

train_generator = datagen.flow_from_dataframe(
    **args,
    dataframe=train_df,
    subset="training",
)

validation_generator = datagen.flow_from_dataframe(
    **args,
    dataframe=train_df,
    subset="validation",
)

if not os.path.exists(c.model_dir):
    os.makedirs(c.model_dir)

vgg16.fit(
    train_generator,
    validation_generator,
    c.img_width,
    c.img_height,
    counts,
    c.model_dir,
    c.batch_size,
    100,
    channel=3,
)
