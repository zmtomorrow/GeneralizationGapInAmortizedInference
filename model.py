import torch
import torch.nn as nn
from network import *
from distributions import *
from tools import *
from torch.distributions.bernoulli import Bernoulli
import numpy as np

class VAE(nn.Module):
    def __init__(self,  opt):
        super().__init__()
        self.z_dim=opt['z_dim']
        self.device=opt['device']
        if opt['data_set']=='BinaryMNIST':
            self.x_flat_dim=784
            self.encoder=densenet_encoder(input_dim=self.x_flat_dim, h_layer_num=opt['h_layer_num'], z_dim=self.z_dim,if_bn=opt['if_bn'])
            self.decoder=densenet_decoder(o_dim=1, h_layer_num=opt['h_layer_num'], z_dim=self.z_dim,if_bn=opt['if_bn'])
            self.criterion  = lambda  data,params : Bernoulli(logits=params).log_prob(data).sum([1,2,3])
            self.sample_op = lambda params: Bernoulli(logits=params).sample()
        elif opt['data_set']=='MNIST':
            self.x_flat_dim=784
            self.out_channels=2
            self.encoder=densenet_encoder(input_dim=self.x_flat_dim, h_layer_num=opt['h_layer_num'], z_dim=self.z_dim,if_bn=opt['if_bn'])
            self.decoder=densenet_decoder(o_dim=2, h_layer_num=opt['h_layer_num'], z_dim=self.z_dim,if_bn=opt['if_bn'])
            self.criterion  = lambda  data,params : batch_logistic_logp(params[:,0:1,:,:],params[:,1:2,:,:],data)
            self.sample_op = lambda params: batch_logistic_sample(params[:,0:1,:,:],params[:,1:2,:,:])

        
        elif opt['data_set'] in ['CIFAR','SVHN']:
            if opt['x_dis']=='MixLogistic':
                if opt['net']=='dc':
                    self.encoder=dc_encoder(z_dim=self.z_dim)
                    self.decoder=dc_decoder(z_dim=self.z_dim,out_channels=100)
                elif opt['net']=='res':
                    self.encoder=res_encoder(z_dim=self.z_dim,res_num=3)
                    self.decoder=res_decoder(z_dim=self.z_dim,out_channels=100)
                
                self.criterion  = lambda  data,params : discretized_mix_logistic_logp(data, params)
                self.sample_op = lambda  params : discretized_mix_logistic_sample(params, 10)
            
            elif opt['x_dis']=='Logistic':
                if opt['net']=='dc':
                    self.encoder=dc_encoder(z_dim=self.z_dim)
                    self.decoder=dc_decoder(z_dim=self.z_dim,out_channels=6)
                elif opt['net']=='res':
                    self.encoder=res_encoder(z_dim=self.z_dim,res_num=3)
                    self.decoder=res_decoder(z_dim=self.z_dim,out_channels=6)

                self.criterion  = lambda  data,params : batch_logistic_logp(params[:,0:3,:,:],params[:,3:6,:,:],data)
                self.sample_op = lambda  params : batch_logistic_sample(params[:,0:3,:,:],params[:,3:6,:,:])
            
            elif opt['x_dis']=='LogisticCAuto':
                self.encoder=res_encoder(z_dim=self.z_dim,res_num=3)
                self.decoder=res_decoder(z_dim=self.z_dim,out_channels=9)
                self.criterion  = lambda  data,params : batch_logistic_autoregressive_logp(params[:,0:3,:,:],params[:,3:6,:,:],params[:,6:9,:,:],data)
                self.sample_op = lambda  params : batch_logistic_mean_transoform_sample(params[:,0:3,:,:],params[:,3:6,:,:],params[:,6:9,:,:])


        self.prior_mu=torch.zeros(self.z_dim, requires_grad=False)
        self.prior_std=torch.ones(self.z_dim, requires_grad=False)
        self.params = list(self.parameters())


    def forward(self, x):
        z_mu, z_std = self.encoder(x)
        eps = torch.randn_like(z_mu).to(self.device)
        zs = eps.mul(z_std).add_(z_mu)
        pxz_params = self.decoder(zs)
        loglikelihood = self.criterion(x, pxz_params)
        kl = batch_KL_diag_gaussian_std(z_mu,z_std,self.prior_mu.to(self.device),self.prior_std.to(self.device))
        elbo = loglikelihood - kl
        return torch.mean(elbo)/np.log(2.)

    def sample(self,num=100):
        with torch.no_grad():
            eps = torch.randn(num,self.z_dim).to(self.device)
            pxz_params = self.decoder(eps)
            return eps,self.sample_op(pxz_params)








    
    
    
    
