# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from common import vgg16,load
import pandas as pd


def predict(
        image_dir_path: str,
        model_dir: str,
        class_num: int,
        img_width: int,
        img_height: int,
        result_file_path: str = 'results.csv',
):
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = datagen.flow_from_directory(
        directory=image_dir_path,
        target_size=(img_width, img_height),
        shuffle=False,
        class_mode="categorical",
    )

    vgg_model = vgg16.create_vgg16_from_latest_weights(
        img_width, img_height, class_num, model_dir, class_num
    )

    loss = vgg_model.predict(test_generator, verbose=1)

    classes = list(test_generator.class_indices.keys())
    prob = pd.DataFrame(loss, columns=classes)
    result = prob.assign(
        predict=prob.idxmax(axis=1),
        actual=[classes[c] for c in test_generator.classes],
        path=test_generator.filenames,
    )
    result.to_csv(result_file_path, index=False)


def predict_from_df(
        csv_path: str,
        image_dir_path: str,
        model_dir: str,
        img_width: int,
        img_height: int,
        x_col: str='',
        y_col: str=''
):
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
    result.to_csv("results.csv", index=False)
