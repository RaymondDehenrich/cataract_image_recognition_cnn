#Used for training new/latest dataset
#flow 
#1. load dataset and call augmentation function.
#2.1 create a temp dataset filled with augmented dataset(even if the image has already been normalized)
#3. Continue/start training using the latest models.
#3.1 Create a new folder named with date and save every 100 epoch checkpoint (depend on runtime, either 10/100/1000 epoch).
from modules.augmentation import normalization_dataset,load_dataset,create_folder
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from modules.cnn import Cnn,load_ckp,save_ckp
torch.manual_seed(21)

def training(net,start_epoch,total_epoch,trainloader,device,optimizer,criterion,epoch_save_rate):
    print(f"[Notice] Beginning Training from epoch [{start_epoch}] to epoch [{total_epoch}]")
    net = net.train()

    for epoch in range(start_epoch,total_epoch):  # loop over the dataset multiple times
        print(f"Epoch [{epoch}]")
        running_loss = 0.0
        num=0
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            num+=1
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs= net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            #print(f"Iteration [{i}]: Loss = {running_loss}\n")
            if (i % 50 == 0 and i!=0):    # print every 2000 mini-batches
                print(f'iter [{i}]| loss: {running_loss / 50:.20f} ')
                running_loss = 0.0
            inputs, labels = data[0].to('cpu'), data[1].to('cpu')
        print(f'Epoch [{epoch}]| loss: {epoch_loss / num:.20f} ')
        epoch_loss=0.0
        print()
        checkpoint = {
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict()
        }
        if (epoch % 2 ==0 and epoch!=0):
            save_ckp(checkpoint,epoch,latest=True)
        if (epoch % epoch_save_rate == 0 and epoch!=0):
            save_ckp(checkpoint,epoch)
    net.to('cpu')
    print('Finished Training')
    return net


def begin_validation(net,testloader,classes,batch_size,device):
    print('Begining Validation')
    net = net.to(device)
    net = net.eval()
    result = {
        'cataract': 0,
        'normal':0,
    }
    prediction = {
        'cataract': 0,
        'normal':0,
    }
    correct = {
        'total_test':0,
        'correct predict':0,
        'cataract': 0,
        'normal':0,
    }
    result_list = ''
    count = 0
    for k, data in enumerate(testloader, 0):
        count=k*10
        images,labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        for i in range(batch_size):
            correct['total_test']+=1
            result[classes[labels[i]]]+=1
            prediction[classes[predicted[i]]]+=1
            if labels[i] == predicted[i]:
                correct[classes[labels[i]]]+=1
        result_list = result_list + ''.join(f'[{j+1+count}. {classes[labels[j]]}|{classes[predicted[j]]}]\n'for j in range(batch_size))
        images,labels = data[0].to('cpu'), data[1].to('cpu')
    for i in classes:
        correct['correct predict']+= correct[i]
    print("Correct Prediction: ",correct)
    print("True Value        : ",result)
    print("Prediction Value  : ",prediction)

    print("Result")
    print(result_list)

    return


def main():
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[Device] Neural Network running on {device}\n')
    #Dataset normalization section
    create_folder('./tmp')
        #THIS IS TEMP
    DIR = "./Dataset/Small/"
    ratio = 218 / 171 #this is fixed
    width = 218 #also fixed
    height = 171 #also fixed
    batch_size=4 #optional
    test_batch_size = 4 #optional
    request_max_epoch = 100
    learning_rate = 0.001 #optional
    epoch_save_rate = 50 #optional
    ckp_path = "./checkpoint/latest_checkpoint.pt" #optional, if no input then will create new checkpoint
    normalization_dataset(DIR,ratio,width,height)


    #Training initialization section
    classes = ('cataract','normal')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])

    #load dataset from tmp
    trainloader = load_dataset('./tmp/dataset_train/',transform,batch_size)
    testloader = load_dataset('./tmp/dataset_val/',transform,test_batch_size,shuffle=False)
    
    #Model initialization
    net = Cnn()
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    total_epoch = request_max_epoch
    start_epoch = 0
    #load latest checkpoint
    #disabled for now
    try:
        net, optimizer, start_epoch = load_ckp(ckp_path, net, optimizer)
        print(f"[Notice] Latest checkpoint loaded [{start_epoch} epoch trained]")
    except:
        print("[Notice] Using new checkpoint")
    #begin training
    net = training(net,start_epoch,total_epoch,trainloader,device,optimizer,criterion,epoch_save_rate)

    begin_validation(net, testloader,classes,test_batch_size,device)


if __name__ == "__main__":
   main()