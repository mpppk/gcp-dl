import vgg16
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime

csv_path = "fashion_mnist.csv"


def select_first_label(tags):
    for label in tags.split(","):
        if label != "train" and label != "test":
            return label


def select_subset(df, subset):
    subset_df = df[df["tags"].str.contains(subset, na=False)].assign(subset=subset)

    return subset_df.assign(label=subset_df["tags"].apply(select_first_label)).dropna()


df = pd.read_csv(csv_path)
train_df = select_subset(df, "train")
test_df = select_subset(df, "test")
print(train_df.head(2))


image_dir_path = "dataset"
datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.25)
counts = train_df["label"].value_counts(normalize=True)

x_col, y_col = "path", "label"
img_width, img_height = 140, 140

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=image_dir_path,
    x_col=x_col,
    y_col=y_col,
    subset="training",
    batch_size=1000,
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
    batch_size=1000,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(img_width, img_height),
)


model_path = "models"
if not os.path.exists(model_path):
    os.makedirs(model_path)

vgg16.fit(
    train_generator,
    validation_generator,
    img_width,
    img_height,
    counts,
    model_path,
    1000,
    100,
)
