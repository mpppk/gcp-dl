# original author: kaityo256
# see https://github.com/kaityo256/fashion_mnist_dump
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

fashion_mnist = keras.datasets.fashion_mnist
train, test = fashion_mnist.load_data()

dirnames = ['Top', 'Trouser', 'Pullover', 'Dress',
            'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']


def save_img_if_does_not_exist(filename, data):
    # if not os.path.isfile(filename):
    #     return
    img = Image.new("L", (28, 28))
    pix = img.load()
    for i in range(28):
        for j in range(28):
            pix[i, j] = int(data[j][i])
    img2 = img.resize((28*5, 28*5))
    img2.save(filename)


def dump(data, dhead):
    for i in range(10):
        dname = "{}/{}".format(dhead, dirnames[i])
        if os.path.isdir(dname) is False:
            os.makedirs(dname)
    images, labels = data
    count = [0]*10
    for i in tqdm(range(len(images))):
        index = labels[i]
        filename = "{}/{}/{}.png".format(dhead, dirnames[index], count[index])
        save_img_if_does_not_exist(filename, images[i])
        count[index] += 1
        # print(filename)

print('downloading test files...')
dump(test, "test")
print('downloading train files...')
dump(train, "train")