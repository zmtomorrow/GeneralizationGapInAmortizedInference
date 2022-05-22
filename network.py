import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x
        
class densenet_encoder(nn.Module):
    def __init__(self,  input_dim=784, h_dim=500, h_layer_num=1, z_dim=50, if_bn= False):
        super().__init__()
        self.h_dim=h_dim
        self.z_dim=z_dim
        self.input_dim=input_dim
        self.h_layer_num=h_layer_num
        self.idenity=Identity()
        self.fc_list=nn.ModuleList([])
        self.bn_list=nn.ModuleList([])
        for i in range(0,h_layer_num+1):
            if i==0:
                self.fc_list.append(nn.Linear(input_dim, self.h_dim))
            else:
                self.fc_list.append(nn.Linear(self.h_dim, self.h_dim))
            if if_bn:
                self.bn_list.append(nn.BatchNorm1d(self.h_dim))
            else:
                self.bn_list.append(self.idenity)

        self.fc_out1 = nn.Linear(self.h_dim, self.z_dim)
        self.fc_out2= nn.Linear(self.h_dim, self.z_dim)

    def forward(self, x, y=None):
        if y is not None:
            x = torch.flatten(x, start_dim=1)
            y = torch.flatten(y, start_dim=1)
            x=torch.cat([x, y], dim=1)
        x=x.view(-1,self.input_dim)
        for i in range(0,self.h_layer_num+1):
            x=F.relu(self.bn_list[i](self.fc_list[i](x)))
        mu=self.fc_out1(x)
        std=torch.nn.functional.softplus(self.fc_out2(x))
        return mu, std
        

class densenet_decoder(nn.Module):
    def __init__(self,o_dim=1,h_dim=500, h_layer_num=1, z_dim=50, if_bn=False):
        super().__init__()
        self.h_dim=h_dim
        self.z_dim=z_dim
        self.o_dim=o_dim
        self.h_layer_num=h_layer_num
        self.idenity=Identity()

        self.fc_list=nn.ModuleList([])
        self.bn_list=nn.ModuleList([])
        for i in range(0,h_layer_num+1):
            if i==0:
                self.fc_list.append(nn.Linear(self.z_dim, self.h_dim))
            else:
                self.fc_list.append(nn.Linear(self.h_dim, self.h_dim))
            if if_bn:
                self.bn_list.append(nn.BatchNorm1d(self.h_dim))
            else:
                self.bn_list.append(self.idenity)
        self.fc_out = nn.Linear(self.h_dim, self.o_dim*784)

    def forward(self,z, y=None):
        if y is not None:
            y = torch.flatten(y, start_dim=1)
            z = torch.cat([y, z], dim=1)
        for i in range (0,self.h_layer_num+1):
            z=F.relu(self.bn_list[i](self.fc_list[i](z)))
        x=self.fc_out(z)
        return x.view(-1,self.o_dim,28,28)





class Classifier(nn.Module):
    def __init__(self, x_dim=784, h_dim=400, y_dim=10, if_bn=True):
        super().__init__()
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, y_dim)
        if if_bn:
            self.bn1 = nn.BatchNorm1d(h_dim)
            self.bn2 = nn.BatchNorm1d(h_dim)
        else:
            self.bn1 = lambda x:x
            self.bn2 = lambda x:x

    def forward(self, x, apply_softmax=True):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x= F.relu(self.bn2(self.fc2(x)))
        logits = self.fc3(x)
        if not apply_softmax:
            return logits
        probs = F.softmax(logits, dim=-1)
        return probs
    

    
    

class Residual(nn.Module):
    def __init__(self, channels):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class fc_encoder(nn.Module):
    def __init__(self, channels=256, latent_channels=64):
        super(fc_encoder, self).__init__()
        self.latent_channels=latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels),
            nn.Conv2d(channels, latent_channels*2, 1)
        )

    def forward(self, x):
        z=self.encoder(x)
        return z[:,:self.latent_channels,:,:].view(x.size(0),-1),F.softplus(z[:,self.latent_channels:,:,:].view(x.size(0),-1))


class fc_decoder(nn.Module):
    def __init__(self, channels=256, latent_channels=64, out_channels=100):
        super(fc_decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d( latent_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, out_channels, 1)
        )

    def forward(self, z):
        # print('here',z.size(0))
        return  self.decoder(z.view(z.size(0),-1,8,8))

