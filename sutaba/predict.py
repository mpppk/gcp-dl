import json

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from common.vgg16 import create_vgg16_from_latest_weights
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
    # print(len(df))
    # df = df[145000:145010]  # for debug
    # print(df.head())

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

    def index_to_class_name(index: int):
        if index == 0:
            return 'sutaba'
        elif index == 1:
            return 'ramen'
        return 'other'

    loss = vgg_model.predict(test_generator, verbose=1)
    for i, probs in enumerate(loss):
        class_name = index_to_class_name(probs.argmax())
        j = json.dumps({'path': test_generator.filenames[i], 'boundingBoxes': [{'tagName': 'predict:' + class_name}]})
        print(j)


if __name__ == '__main__':
    base_dir = '/mnt/disks/sutaba/old/'
    predict_from_df(
        # csv_path='20210228assets.csv',
        csv_path=base_dir + 'asset_csv/20210228assets.csv',
        # image_dir_path='/Users/yuki/ghq/github.com/mpppk/twitter/images',
        image_dir_path='/mnt/disks/sutaba/images',
        # model_dir='/Users/yuki/ghq/github.com/mpppk/gcp-dl/sutaba/old_models',
        model_dir=base_dir + 'old_models',
        img_width=225,
        img_height=225,
        x_col='path',
        y_col='label',
    )
