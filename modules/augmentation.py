from PIL import Image, ImageOps
import os
import shutil
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
#this file is doing normalization & duplicate image such as flip and mirror for increase dataset size
#it will normalize the image to monochrome, and to the specified ratio, width and height.

#flow
#dataset udh dlm 2 folder(test dan val), isinya udh ada 3 atau lebih folder(label) yang berisi image

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_dataset(dir,transform,batch_size,shuffle=True):
    dataset = torchvision.datasets.ImageFolder(root=dir,transform=transform)
    datasetloader = DataLoader(dataset, batch_size=batch_size,shuffle=shuffle, num_workers=2)
    return datasetloader

def create_folder(dir):
    try:
        os.mkdir(dir)
        return
    except:
        shutil.rmtree(dir)
        os.mkdir(dir)
        return

def crop_image(image,ratio = 2464 / 1632,width = 2464 ,height = 1632):
    current_ratio = image.width / image.height
    if current_ratio > ratio:
        new_width = int(image.height * ratio)
        left = (image.width - new_width) // 2
        right = left + new_width
        top, bottom = 0, image.height
    else:
        new_height = int(image.width / ratio)
        top = (image.height - new_height) // 2
        bottom = top + new_height
        left, right = 0, image.width
    cropped_image = image.crop((left, top, right, bottom))
    resized_image = cropped_image.resize((width, height))
    return resized_image

def mirror_and_flip_image(image):
    mirror_image = ImageOps.mirror(image)
    flipped_image = ImageOps.flip(image)
    flipped_mirror_image = ImageOps.flip(image)
    flipped_mirror_image = ImageOps.mirror(flipped_mirror_image)
    return mirror_image,flipped_image,flipped_mirror_image


def normalization_dataset(dataset_dir,ratio,width,height):
    #create root temp dataset folder
    print("[Notice] Normalizing Dataset")
    create_folder('./tmp/dataset_train')
    create_folder('./tmp/dataset_val')
    #untuk train
    for label in os.listdir(os.path.join(dataset_dir,'train')):
        create_folder(f'./tmp/dataset_train/{label}')
        count=0
        for image_list in os.listdir(os.path.join(dataset_dir,f'train/{label}')):
            image_dir = os.path.join(dataset_dir,f'train/{label}/{image_list}')
            image = Image.open(image_dir)
            cropped_image = crop_image(image,ratio,width,height)
            mirror_image, flipped_image, mirrorflip_image = mirror_and_flip_image(cropped_image)
            cropped_image.save(os.path.join(f'./tmp/dataset_train/{label}',f"{count}.png"))
            mirror_image.save(os.path.join(f'./tmp/dataset_train/{label}',f"{count+1}.png"))
            flipped_image.save(os.path.join(f'./tmp/dataset_train/{label}',f"{count+2}.png"))
            mirrorflip_image.save(os.path.join(f'./tmp/dataset_train/{label}',f"{count+3}.png"))
            count=count+4

    #untuk val
    for label in os.listdir(os.path.join(dataset_dir,'val')):
        create_folder(f'./tmp/dataset_val/{label}')
        count=0
        for image_list in os.listdir(os.path.join(dataset_dir,f'val/{label}')):
            image_dir = os.path.join(dataset_dir,f'val/{label}/{image_list}')
            image = Image.open(image_dir)
            cropped_image = crop_image(image,ratio,width,height)
            cropped_image.save(os.path.join(f'./tmp/dataset_val/{label}',f"{count}.png"))
            count=count+1
  
    return

def normalization_input(input_dir,ratio = 1024 / 1512,width = 256 ,height = 378):
    tmp_folder = './tmp/input_normalized'
    buffer_folder = './tmp/input_normalized/buffer'
    create_folder(tmp_folder)
    create_folder(buffer_folder)
    count=0
    for image_list in os.listdir(input_dir):
        image_dir = os.path.join(input_dir,image_list)
        image = Image.open(image_dir)
        cropped_image = crop_image(image,ratio,width,height)
        cropped_image.save(os.path.join(buffer_folder,f"{count}.png"))
        count=count+1