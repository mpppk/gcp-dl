import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import load
import vgg16
import pandas as pd

img_width, img_height = 56, 56
model_dir = "buckets/mpppk-fashion-mnist/models"
image_dir_path = "dataset"
csv_path = "fashion_mnist.csv"
x_col, y_col = "path", "label"

datagen = ImageDataGenerator(rescale=1.0 / 255)

train_df, test_df = load.load(csv_path)
counts = test_df["label"].value_counts(normalize=True)

test_generator = datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=image_dir_path,
    target_size=(img_width, img_height),
    x_col=x_col,
    y_col=y_col,
    shuffle=False,
    class_mode="categorical",
)

vgg_model = vgg16.create_vgg16_from_latest_weights(
    img_width, img_height, len(counts.keys()), model_dir, 3
)

loss = vgg_model.predict(test_generator, verbose=1)

classes = list(test_generator.class_indices.keys())
prob = pd.DataFrame(loss, columns=classes)
result = prob.assign(
    predict=prob.idxmax(axis=1),
    actual=[classes[c] for c in test_generator.classes],
    path=test_generator.filenames,
)
result.to_csv("results.csv")