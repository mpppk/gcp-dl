#! /bin/bash

bash -c "$(curl -L dd.nibo.sh)"
git clone https://github.com/mpppk/gcp-dl
./gcp-dl/fashion_mnist/download_fashion_mnist.py
./gcp-dl/fashion_mnist/install_gcsfuse.sh
./gcp-dl/fashion_mnist/mount_bucket.sh