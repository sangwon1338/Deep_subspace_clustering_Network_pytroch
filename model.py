import torch
from torch import nn
import torch.nn.init as init
from tensorly.decomposition import tucker
import tensorly as tl
from tensorly.tucker_tensor import tucker_to_tensor
from torch.autograd import Variable
import numpy as np
torch.manual_seed(0)
tl.set_backend('pytorch')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def init_weight(m):
    if type(m)==(nn.Conv2d or nn.ConvTranspose2d):
        init.xavier_normal(m.weight)
        m.bias.data.fill_(0)

class COil20_Network(nn.Module):
    def __init__(self,test=False):
        super(COil20_Network, self).__init__()

        self.encoder = nn.Conv2d(1,15,3,stride=2,padding=1)
        
        self.decoder =nn.ConvTranspose2d(15, 1, 3, stride=2,padding=1)
        self.Relu= nn.ReLU()
        
        
        self.test=test
        self.shape=[]
        
        self.Coef = nn.Parameter(tl.tensor(1e-4*tl.ones((720,720,4))))

    def forward(self, x):
       self.shape.append(x.shape)
       
       if self.test == True:

            x = self.encoder(x)
            x = self.Relu(x)
            self.shape.append(x.shape)
            z_conv,z_ssc = self.self_expressive(x)
            x = torch.reshape(z_ssc,self.shape[1])
            x = self.decoder(x,output_size=self.shape[0])
            x = self.Relu(x)
            return x ,z_conv,z_ssc,self.Coef

       else: 

            x = self.encoder(x)
            x = self.Relu(x)
            print(1)
            core ,factors = tucker(self.Coef,rank =[360,360,2])
            print(core.shape)
            Coef = tucker_to_tensor(core,factors)
            print(Coef.shape)
            x = self.decoder(x,output_size=self.shape[0])
            x = self.Relu(x)
        
            return  x

    def self_expressive(self,x):
        z_conv = x.view(x.shape[0],-1)
        z_ssc = torch.matmul(self.Coef, z_conv)
        return z_conv,z_ssc  