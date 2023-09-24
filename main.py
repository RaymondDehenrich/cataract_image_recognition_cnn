#used for predicting the image
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from modules.cnn import Cnn,load_ckp
from modules.augmentation import normalization_input,create_folder,load_dataset
import os
import shutil
import numpy as np
torch.manual_seed(1)

def check_output(output_dir):
   if(len(os.listdir(output_dir))> 0):
      print('[Warning] the output folder is not empty, please backput the output!')
      print(f'[Warning] Note, you will be deleting a total of {len(os.listdir(output_dir))} file!')
      print('[Warning] Delete output file? (y/n)')
      inputs = input('>>')
      if inputs == "Y" or inputs == "y":
         create_folder(output_dir)
         return
      else:
         print('[Notice] Stopping system!')
         exit()


def begin_classification(model,data,device):
   model = model.eval()
   inputs = data[0].to(device)
   outputs = model(inputs)
   outputs = outputs.to('cpu')
   _, true_outputs = torch.max(outputs, 1)
   return true_outputs

def main():
   device ='cuda' if torch.cuda.is_available() else 'cpu'
   print(f'[Device] Neural Network running on {device}\n')
   input_dir = './input'
   output_dir = './output'
   check_output(output_dir)
   ckp_path = './checkpoint/latest_checkpoint.pt'
   batch_size = 1
   classes = ('cataract','normal')
   create_folder('./tmp')
   transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
   print('[Notice] Normalizing Input')
   normalization_input(input_dir,ratio=218 / 171,width=218,height=171)
   model = Cnn()
   optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)#not important, used so can use load_ckp function
   print('[Notice] Loading model checkpoint')
   model, _, epoch_done = load_ckp(ckp_path, model, optimizer)
   print(f'[Notice] Loaded model with {epoch_done} epochs')
   model = model.to(device)
   print()
   print(f'[Notice] Begining prediction')
   inputloader = load_dataset('./tmp/input_normalized',transform=transform,batch_size=batch_size,shuffle=False)
   for k, data in enumerate(inputloader,0):
      outputs = begin_classification(model,data,device)
      image_list = os.listdir(input_dir)
      src_img = os.path.join(input_dir,image_list[k])
      dst_img = os.path.join(output_dir,f"{k}-{classes[outputs]}.png")
      shutil.copy(src_img,dst_img)
   print(f'[Notice] Prediction complete, please check the output folder!')
   create_folder('./tmp')


if __name__ == "__main__":
   main()

