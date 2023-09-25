# How to run
create a new conda environment (optional)<br />
install pip in conda<br />
run 'pip install -r requirements.txt'<br />
Note: please install torch 2.0.1 (torchvision included) according to your need from [Pytorch](https://pytorch.org/)<br />
you will get a line like : pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117<br />
install torch using that line<br />
if you would like pre-trained models, go to [Hugging Face](https://huggingface.co/RaymondDehenrich/cataract_image_recognition_cnn/tree/main)<br />

to train, use the train.py with folder structure of './Dataset/Small/train/<strong>class</strong>/<strong>image</strong>' and './Dataset/Small/val/<strong>class</strong>/<strong>image</strong>'<br />
to predict, use the main.py with folder structure of './input/<strong>image</strong>'. output will come out in ./output/</strong>num</strong>-<strong>class</strong>.png<br />

Please note that the image will be resized and cropped to 218 x 171 for training and prediction, so make sure to capture the object in the middle.<br />
Dataset Source: [Kaggle Dataset by Nandan Padia](https://www.kaggle.com/datasets/nandanp6/cataract-image-dataset)
