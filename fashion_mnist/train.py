from common import training


if __name__ == '__main__':
    training.train(
        csv_path='fashion_mnist.csv',
        image_dir_path='dataset',
        model_path='models',
        # img_widthが28だとなんかエラーで落ちるので雑に2倍する
        img_width=56,
        img_height=56,
        batch_size=1024,
        validation_split=0.25
    )

