# How to run
# create a new conda environment (optional)
# install pip in conda
# run 'pip install -r requirements.txt'
# Note: please install torch 2.0.1 (torchvision included) according to your need from https://pytorch.org/
# you will get a line like : pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# install torch using that line
# if you would like pre-trained models, go to https://huggingface.co/CrispySalt/cataract_image_recognition_cnn/tree/main
# to train, use the train.py with folder structure of './Dataset/Small/train/<class>/<image>' and './Dataset/Small/val/<class>/<image>'
# to predict, use the main.py with folder structure of './input/<image>'. output will come out in ./output/<num>-<class>.png
# Please note that the image will be resized to 218 x 171 for training and prediction, so make sure to capture the object in the middle.