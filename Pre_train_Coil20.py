import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from model import *
from utill import COIL20
from torchvision import datasets, transforms
import argparse
from torch.utils.data import Dataset
import numpy as np
import tensorly as tl
tl.set_backend('pytorch')
import torch.nn.init as init
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
parser = argparse.ArgumentParser(description='PreTrain DSC')
parser.add_argument('--batch' ,type=int, default=256, metavar='N')
parser.add_argument('--dataset', type=str)
parser.add_argument('--num_class', type=int, default=20,metavar='N')
parser.add_argument('--epoch', type=int, default=100,metavar='N')
parser.add_argument('--lr',type=int, default=1e-3,metavar='N')


args = parser.parse_args()
transform = transforms.Compose([transforms.ToTensor(), ])

data_path = './data/COIL20.mat'

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = COil20_Network().to(device)

model.apply(init_weight)



train_dataset=COIL20(transform=transform,path=data_path)
trainloader=DataLoader(train_dataset,batch_size=args.batch,shuffle=True)


is_best=0
best_acc=0
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler=optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30,gamma=0.95)
criterion=nn.MSELoss(reduction="sum")

for epoch in range(args.epoch):
    model.train()

    running_loss = 0.0


    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, _ = data

        inputs = inputs.to(device)
        
        optimizer.zero_grad()
        outputs= model(inputs)
       
        loss = criterion(outputs, inputs)


        loss.backward()
        optimizer.step()
        running_loss += loss.item()
       
   # print("epoch : {} , loss: {:.8f}".format(epoch + 1, running_loss / len(trainloader)))
    

torch.save(model.state_dict(),'./model_weight/Coil20_pre{}.pth'.format(args.epoch))



