import torch
import torch.nn as nn
from network import *
from distributions import *
from utils import  batch_KL_diag_gaussian_std
from torch.distributions.bernoulli import Bernoulli
import numpy as np
import torch.nn.functional as F

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
            if opt['x_dis']=='Logistic':
                self.out_channels=2
                self.encoder=densenet_encoder(input_dim=self.x_flat_dim, h_layer_num=opt['h_layer_num'], z_dim=self.z_dim,if_bn=opt['if_bn'])
                self.decoder=densenet_decoder(o_dim=2, h_layer_num=opt['h_layer_num'], z_dim=self.z_dim,if_bn=opt['if_bn'])
                self.criterion  = lambda  data,params : discretized_logistic(params[:,0:1,:,:],params[:,1:2,:,:],data)
                self.sample_op = lambda params: discretized_logistic_sample(params[:,0:1,:,:],params[:,1:2,:,:])
            else: 
                raise NotImplementedError
        
        elif opt['data_set'] in ['CIFAR','SVHN']:
            self.encoder=fc_encoder(latent_channels=opt['z_channels'])
            if opt['x_dis']=='MixLogistic':
                self.decoder=fc_decoder(latent_channels=opt['z_channels'],out_channels=100)       
                self.criterion  = lambda  data,params :discretized_mix_logistic_uniform(data, params)
                self.sample_op = lambda  params : discretized_mix_logistic_sample(params)
            elif opt['x_dis']=='Logistic':
                self.decoder=fc_decoder(latent_channels=opt['z_channels'],out_channels=9)       
                self.criterion  = lambda  data,params :discretized_logistic(data, params)
                self.sample_op = lambda params: discretized_logistic_sample(params)

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
    
    def denoising_forward(self, tilde_x,x):
        z_mu, z_std = self.encoder(tilde_x)
        eps = torch.randn_like(z_mu).to(self.device)
        zs = eps.mul(z_std).add_(z_mu)
        pxz_params = self.decoder(zs)
        loglikelihood = self.criterion(x, pxz_params)
        kl = batch_KL_diag_gaussian_std(z_mu,z_std,self.prior_mu.to(self.device),self.prior_std.to(self.device))
        elbo = loglikelihood - kl
        return torch.mean(elbo)/np.log(2.)
    
    def posterior_forward(self, params,x):
        z_mu, z_std = params[0],F.softplus(params[1])
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








    
    
    
    
