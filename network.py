import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import *



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


class dc_encoder(nn.Module):
    def __init__(self,z_dim,if_label=False,y_dim=10):
        super().__init__()
        self.z_dim=z_dim
        self.if_label=if_label
        self.y_dim=y_dim
        self.activation=nn.LeakyReLU(0.2)
        self.conv1=nn.Conv2d(3, 32, 5, 2, 2)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32, 64, 5, 2, 2)
        self.bn2=nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(64, 128, 5, 2, 2)
        self.bn3=nn.BatchNorm2d(128)
        self.conv4=nn.Conv2d(128, 256, 5, 2, 2)
        self.bn4=nn.BatchNorm2d(256)
        if if_label:
            self.y_linear=nn.Linear(10, 20)
            self.label_bn=nn.BatchNorm1d(20)
            # self.cat_forward
            self.ff=nn.Linear(1024+20, 256)
            self.ff_bn=nn.BatchNorm1d(256)
            self.en_mean=nn.Linear(256, self.z_dim)
            self.en_logstd=nn.Linear(256, self.z_dim)
        else:
            self.ff=nn.Linear(1024, 256)
            self.ff_bn=nn.BatchNorm1d(256)
            self.en_mean=nn.Linear(256, self.z_dim)
            self.en_logstd=nn.Linear(256, self.z_dim)
        
    
    def forward(self,x, y=None):
        x=x.view(-1,3,32,32)
        x=self.activation(self.bn1(self.conv1(x)))
        x=self.activation(self.bn2(self.conv2(x)))
        x=self.activation(self.bn3(self.conv3(x)))
        x=self.activation(self.bn4(self.conv4(x)))

        x=torch.flatten(x, start_dim=1)
        if self.if_label:
            y = y.view(-1, self.y_dim).type(torch.float32)
            y_embedding=self.activation(self.label_bn(self.y_linear(y)))
            xy=self.activation(self.ff_bn(self.ff(torch.cat([x,y_embedding],1))))
            mu=self.en_mean(xy)
            std=F.softplus(self.en_logstd(xy))
            return mu,std
        else:
            x=self.activation(self.ff_bn(self.ff(x)))
            mu=self.en_mean(x)
            std=F.softplus(self.en_logstd(x))
            return mu,std



class dc_decoder(nn.Module):
    def __init__(self,z_dim,out_channels=100,if_label=False,y_dim=10):
        super().__init__()
        if if_label:
            self.z_dim=z_dim+y_dim
        else:
            self.z_dim=z_dim
        self.out_channels=out_channels
        self.ff1 = nn.Linear(self.z_dim, 512)
        self.bn1= nn.BatchNorm1d(512)
        self.ff2 = nn.Linear(512, 1024)
        self.bn2= nn.BatchNorm1d(1024)
        self.activation=nn.LeakyReLU(0.2)
        self.deconv1=nn.ConvTranspose2d(256, 128, 5, 1)
        self.bn3=nn.BatchNorm2d(128)
        self.deconv2=nn.ConvTranspose2d(128, 64, 5, 1)
        self.bn4=nn.BatchNorm2d(64)
        self.deconv3=nn.ConvTranspose2d(64, 64, 5, 1)
        self.bn5=nn.BatchNorm2d(64)
        self.deconv4=nn.ConvTranspose2d(64, out_channels, 6, 2)
    
    def forward(self,z,y=None):
        if y is not None:
            z = torch.cat([y, z], dim=1)
        x=self.activation(self.bn1(self.ff1(z)))
        x=self.activation(self.bn2(self.ff2(x))).view(-1,256,2,2)
        x=self.activation(self.bn3(self.deconv1(x)))
        x=self.activation(self.bn4(self.deconv2(x)))
        x=self.activation(self.bn5(self.deconv3(x)))
        x=self.deconv4(x)
        return x




class res_encoder(nn.Module):
    def __init__(self,z_dim, res_num=3, h_dim=400, if_label=False,y_dim=10):
        super().__init__()
        self.z_dim=z_dim
        self.h_dim=h_dim
        self.y_dim=y_dim
        self.res_num=res_num
        self.activation=nn.ELU()
        self.if_label=if_label
        self.resnet_cnn00=nn.Conv2d(3,h_dim,3,1,1)
        self.resnet_cnn01=nn.Conv2d(3,h_dim,3,1,1)

        self.resnet_cnn=torch.nn.ModuleList([nn.Conv2d(h_dim,h_dim,3,1,1) for i in range(0,res_num*2-1)])
        self.resnet_batchnorm=torch.nn.ModuleList([nn.BatchNorm2d(h_dim) for i in range(0,res_num*2-1)])
        
        self.batchnorm_out=nn.BatchNorm2d(h_dim)
        if if_label:
            self.y_linear=nn.Linear(y_dim, 20)
            self.label_bn=nn.BatchNorm1d(20)
            self.en_mean = nn.Linear(h_dim*4*4+20, self.z_dim)
            self.en_logstd = nn.Linear(h_dim*4*4+20, self.z_dim)
        else:
            self.en_mean = nn.Linear(h_dim*4*4, self.z_dim)
            self.en_logstd = nn.Linear(h_dim*4*4, self.z_dim)


    def forward(self,x, y=None):
        x=x.view(-1,3,32,32)
        batch_size=x.size(0)
        for i in range(0, self.res_num):
            if i==0:
                x_mid=self.resnet_cnn00(x)
                x=self.resnet_cnn01(x)
            else:
                x_mid=self.resnet_batchnorm[i*2-1](x_mid)
                x_mid=self.activation(x_mid)
                x_mid=self.resnet_cnn[i*2-1](x_mid)
            x_mid=F.avg_pool2d(x_mid,[2,2])
            x_mid=self.resnet_batchnorm[i*2](x_mid)
            x_mid=self.activation(x_mid)
            x_mid=self.resnet_cnn[i*2](x_mid)
            x=F.avg_pool2d(x,[2,2])
            x=x+x_mid
        x=self.batchnorm_out(x)
        x=self.activation(x)
        x=x.view(batch_size,-1)
        if self.if_label:
            y = y.view(-1, self.y_dim).type(torch.float32)
            y_embedding=self.activation(self.label_bn(self.y_linear(y)))
            mu=self.en_mean(torch.cat([x,y_embedding],1))
            std=F.softplus(self.en_logstd(torch.cat([x,y_embedding],1)))
            return mu,std
        else:
            mu=self.en_mean(x)
            std=F.softplus(self.en_logstd(x))
            return mu,std









class res_encoder(nn.Module):
    def __init__(self,z_dim, res_num=3, h_dim=400):
        super().__init__()
        self.z_dim=z_dim
        self.h_dim=h_dim
        self.res_num=res_num
        self.activation=nn.ELU()
        self.resnet_cnn00=nn.Conv2d(3,h_dim,3,1,1)
        self.resnet_cnn01=nn.Conv2d(3,h_dim,3,1,1)

        self.resnet_cnn=torch.nn.ModuleList([nn.Conv2d(h_dim,h_dim,3,1,1) for i in range(0,res_num*2-1)])
        self.resnet_batchnorm=torch.nn.ModuleList([nn.BatchNorm2d(h_dim) for i in range(0,res_num*2-1)])
        
        self.batchnorm_out=nn.BatchNorm2d(h_dim)

        self.en_mean = nn.Linear(h_dim*4*4, self.z_dim)
        self.en_logstd = nn.Linear(h_dim*4*4, self.z_dim)


    def forward(self,x, y=None):
        batch_size=x.size(0)
        for i in range(0, self.res_num):
            if i==0:
                x_mid=self.resnet_cnn00(x)
                x=self.resnet_cnn01(x)
            else:
                x_mid=self.resnet_batchnorm[i*2-1](x_mid)
                x_mid=self.activation(x_mid)
                x_mid=self.resnet_cnn[i*2-1](x_mid)
            x_mid=F.avg_pool2d(x_mid,[2,2])
            x_mid=self.resnet_batchnorm[i*2](x_mid)
            x_mid=self.activation(x_mid)
            x_mid=self.resnet_cnn[i*2](x_mid)
            x=F.avg_pool2d(x,[2,2])
            x=x+x_mid
        x=self.batchnorm_out(x)
        x=self.activation(x)
        x=x.view(batch_size,-1)
        mu=self.en_mean(x)
        std=F.softplus(self.en_logstd(x))
        return mu,std

class res_decoder(nn.Module):
    def __init__(self,z_dim, out_channels=100, res_num=3, h_dim=400):
        super().__init__()

        self.z_dim=z_dim
        self.h_dim=h_dim
        self.out_channels=out_channels
        self.res_num=res_num
        self.fc = nn.Linear(self.z_dim, self.h_dim*4*4)
        self.activation=nn.ELU()

        self.resnet_up=torch.nn.ModuleList([nn.ConvTranspose2d(h_dim,h_dim, 4, 2, 1) for i in range(0,res_num*2)])
    
        self.resnet_cnn1=torch.nn.ModuleList([nn.Conv2d(h_dim,h_dim,1,1) for i in range(0,res_num)])

        self.resnet_cnn=torch.nn.ModuleList([nn.Conv2d(h_dim,h_dim,5,1,2) for i in range(0,res_num)])
        self.resnet_batchnorm=torch.nn.ModuleList([nn.BatchNorm2d(h_dim) for i in range(0,res_num*3)])
        
        self.batchnorm_out=nn.BatchNorm2d(h_dim)
        self.cnn_out=nn.Conv2d(h_dim,out_channels,3,1,1)

    def forward(self,z):
        x=self.fc(z).view(-1,self.h_dim,4,4)
        for i in range(0, self.res_num):
            x_mid=self.resnet_batchnorm[i*3](x)
            x_mid=self.activation(x_mid)
            x_mid=self.resnet_up[i*2](x_mid)
            x_mid=self.resnet_batchnorm[i*3+1](x_mid)
            x_mid=self.activation(x_mid)
            x_mid=self.resnet_cnn[i](x_mid)
            x_mid=self.resnet_batchnorm[i*3+2](x_mid)
            x_mid=self.activation(x_mid)
            x_mid=self.resnet_cnn1[i](x_mid)
            x=self.resnet_up[i*2+1](x)
            x=x+x_mid
        x=self.batchnorm_out(x)
        x=self.activation(x)
        x=self.cnn_out(x)
        return x



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
    def __init__(self, channels=256, latent_dim=8, embedding_dim=32):
        super(fc_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels),
            nn.Conv2d(channels, latent_dim * embedding_dim*2, 1)
        )

    def forward(self, x):
        z=self.encoder(x)
        return z[:,:256,:,:].view(-1,16384),F.softplus(z[:,256:,:,:].view(-1,16384))


class fc_decoder(nn.Module):
    def __init__(self, channels=256, latent_dim=8, embedding_dim=32, out_channels=100):
        super(fc_decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim * embedding_dim, channels, 1, bias=False),
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
        return  self.decoder(z.view(-1,256,8,8))
