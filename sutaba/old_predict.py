from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from common.vgg16 import create_vgg16_from_latest_weights


def predict_from_df(
        csv_path: str,
        image_dir_path: str,
        model_dir: str,
        img_width: int,
        img_height: int,
        x_col: str = 'path',
        y_col: str = 'label'
):
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    df = pd.read_csv(csv_path)
    df = df[~df["tags"].str.contains('invalid', na=False)]
    # df = df[:100]  # for debug
    print(df.head())
    # counts = df[y_col].value_counts(normalize=True)

    test_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=image_dir_path,
        target_size=(img_width, img_height),
        x_col=x_col,
        y_col=x_col,  # FIXME
        shuffle=False,
        class_mode="categorical",
    )

    vgg_model = create_vgg16_from_latest_weights(
        img_width, img_height, 3, model_dir, 3
    )

    loss = vgg_model.predict(test_generator, verbose=1)

    classes = ['sutaba', 'ramen', 'other']
    prob = pd.DataFrame(loss, columns=classes)
    result = prob.assign(
        predict=prob.idxmax(axis=1),
        path=test_generator.filenames,
    )
    result.to_csv("results.csv", index=False)


if __name__ == '__main__':
    predict_from_df(
        csv_path='20210228assets.csv',
        image_dir_path='/Users/yuki/ghq/github.com/mpppk/twitter/images',
        model_dir='old_models',
        img_width=225,
        img_height=225,
        x_col='path',
        y_col='label',
    )
