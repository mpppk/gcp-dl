# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from common import vgg16, load
import config
import pandas as pd

c = config.Config()

datagen = ImageDataGenerator(rescale=1.0 / 255)

train_df, test_df = load.load(c.csv_path)
counts = test_df["label"].value_counts(normalize=True)

test_generator = datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=c.image_dir_path,
    target_size=(c.img_width, c.img_height),
    x_col=c.x_col,
    y_col=c.y_col,
    shuffle=False,
    class_mode="categorical",
)

vgg_model = vgg16.create_vgg16_from_latest_weights(
    c.img_width, c.img_height, len(counts.keys()), c.model_dir, 3
)

loss = vgg_model.predict(test_generator, verbose=1)

classes = list(test_generator.class_indices.keys())
prob = pd.DataFrame(loss, columns=classes)
result = prob.assign(
    predict=prob.idxmax(axis=1),
    actual=[classes[c] for c in test_generator.classes],
    path=test_generator.filenames,
)
result.to_csv("results.csv", index=False)