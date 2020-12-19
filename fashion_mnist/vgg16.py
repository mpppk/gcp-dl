from glob import glob
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Activation,
    Conv2D,
    Flatten,
    Dropout,
    MaxPooling2D,
    BatchNormalization,
)
from keras.applications.vgg16 import VGG16
from keras import regularizers, optimizers
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import os, datetime


def create_vgg16(
    img_width: int, img_height: int, class_num: int, lr=1e-3, momentum=0.9
):
    input_tensor = Input(shape=(img_width, img_height, 3))
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation="relu"))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(class_num, activation="softmax"))

    vgg_model = Model(vgg16.input, top_model(vgg16.output))

    for layer in vgg_model.layers[:15]:
        layer.trainable = False

    vgg_model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.SGD(lr=lr, momentum=momentum),
        metrics=["accuracy"],
    )
    return vgg_model


def get_latest_modified_file_path(dirname: str):
    target = os.path.join(dirname, "*")
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    if len(files) == 0:
        return None
    latest_modified_file_path = sorted(files, key=lambda files: files[1])[-1]
    return latest_modified_file_path[0]


def create_vgg16_from_weights(
    img_width: int, img_height: int, class_num, weights_path: str
):
    vgg16_model = create_vgg16(img_width, img_height, class_num)
    p = get_latest_modified_file_path(weights_path)
    if p is not None:
        print("model is loaded from " + p)
        vgg16_model.load_weights(p)

    return vgg16_model


def fit(
    train_generator,
    validation_generator,
    img_width: int,
    img_height: int,
    counts: int,
    model_path: str,
    batch_size: int,
    epochs: int,
):
    vgg_model = create_vgg16_from_weights(
        img_width, img_height, len(counts.keys()), model_path
    )

    n = datetime.datetime.now()
    nstr = f"{n.year}-{n.month:02}-{n.day:02}_{n.hour:02}-{n.minute:02}-{n.second:02}"
    fpath = (
        f"{model_path}/{nstr}" + "weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5"
    )
    cp_cb = keras.callbacks.ModelCheckpoint(
        filepath=fpath, monitor="val_loss", verbose=1, save_best_only=True, mode="auto"
    )
    tf_callback = TensorBoard(log_dir="logs", histogram_freq=1)
    callbacks = [cp_cb, tf_callback]

    class_weight = {
        i: counts[name] for name, i in train_generator.class_indices.items()
    }

    return vgg_model.fit(
        train_generator,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weight,
        batch_size=batch_size,
        epochs=epochs,
    )