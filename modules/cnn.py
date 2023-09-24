import torch
import torch.nn as nn
import torch.optim as optim
import os
torch.manual_seed(1)


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()

        self.convolution= nn.Sequential(
        nn.Conv2d(1, 2, 7,padding=3),
        nn.Conv2d(2, 8, 7,padding=3),
        nn.Conv2d(8, 10, 5,padding=2),
        nn.Conv2d(10, 12, 3,padding=1),
        nn.Conv2d(12, 14, 3,padding=1),
        nn.MaxPool2d(4, 4),
        nn.Conv2d(14, 16, 3,padding=1),
        nn.MaxPool2d(2, 2)
        )

        self.fully_connected_layer=nn.Sequential(
        nn.Linear(9072, 12000),
        nn.Dropout(0.2),
        nn.Linear(12000, 9000),
        nn.Linear(9000, 3000),
        nn.Linear(3000, 1500),
        nn.Dropout(0.2),
        nn.Linear(1500, 3000),
        nn.Linear(3000, 3500),
        nn.ReLU()
        )
        self.output_layer=nn.Sequential(
            nn.Linear(3500, 2)
        )

    def forward(self, x):
        image_conv = self.convolution(x)
        image_conv = torch.flatten(image_conv,1)
        output = self.fully_connected_layer(image_conv)
        output = self.output_layer(output)
        return output
    

#218/4 = 616/2 = 28
#171/4 = 408/2 = 22
def save_ckp(state,epoch,checkpoint_dir='./checkpoint/',latest=False):
    if latest == False:
        f_path = os.path.join(checkpoint_dir,f'checkpoint_epoch_{epoch}.pt')
    else: f_path = os.path.join(checkpoint_dir,f'latest_checkpoint.pt')
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']