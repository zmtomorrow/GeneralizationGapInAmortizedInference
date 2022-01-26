from scipy.stats import norm
import numpy as np
from tools import *
from utils import *
from scipy.special import expit as sigmoid
import torch.nn as nn
from coder.rANS import *
import torch.nn.functional as F


def optimal_posterior(img,model,opt):
    torch.manual_seed(0)
    np.random.seed(0)

    with torch.no_grad():
        mean,std = model.encoder(torch.tensor(img))
        softplusinv_std=softplusinv(std)
        init_post_params=(mean,softplusinv_std)
        post_params = [nn.Parameter(post_param.data).to(opt['device']) for post_param in init_post_params]
    optimizer = torch.optim.Adam(post_params, lr=opt['optimal_lr'])    

    for i in range(opt['iterations']):
        optimizer.zero_grad()
        if opt['data_set']=='CIFAR':
            L = -model.posterior_forward(post_params,torch.tensor(img).view(1,3,32,32).to(opt['device']))
        else:
            L = -model.posterior_forward(post_params,torch.tensor(img).view(1,1,28,28).to(opt['device']))
        L.backward()
        optimizer.step()
        
    mean=post_params[0].cpu().detach().view(-1).numpy()
    std=F.softplus(post_params[1]).cpu().detach().view(-1).numpy()+1e-7
    return mean,std



def BBCompression(img,ansstack,model,opt):
    model.encoder.eval()
    model.decoder.eval()
    encoder=model.encoder
    decoder=model.decoder

    discretization_scheme =  opt['discretization_scheme']
    discretization_centres = opt['discretization_centres']
        

    sample_list=[]
    index_list=[]

    if opt['optimal']==False:
        with torch.no_grad():
            mean,std=encoder(torch.tensor(img,dtype=torch.float32).to(opt['device']))
        mean=mean.cpu().numpy()[0]
        std=std.cpu().numpy()[0]+1e-7
    else:
        mean,std=optimal_posterior(img,model,opt)


    for i in range(0,opt['z_dim']):
        s_bar=ansstack.pop()
        x=discrete_gaussian_ppf(discretization_scheme,mean[i],std[i],s_bar,opt['p_prec'])
        cdf_min,p =discrete_gaussian_stats(discretization_scheme,mean[i],std[i],x,opt['p_prec'])
        ansstack.update(s_bar,cdf_min,p)
        index_list.append(x)
        sample_list.append(discretization_centres[x])
    z_sample=np.asarray(sample_list).reshape(1,-1)

    with torch.no_grad():
        x_p=decoder(torch.tensor(z_sample,dtype=torch.float32).to(opt['device'])).cpu().numpy()

    if opt['obs_dis']=='LogisticCA':
        c_num=3
        all_means=x_p[:,:c_num,:,:].reshape(c_num,-1)
        log_scales=x_p[:,c_num:2*c_num,:,:].reshape(c_num,-1)
        mean_linear=x_p[:,2*c_num:3*c_num,:,:].reshape(c_num,-1)
        
        img=img.reshape(c_num,-1)
        remain_dim=img.shape[1]

        for c in range(c_num-1,-1,-1):
            if c==0:
                mean=all_means[0]
            elif c==1:
                mean=all_means[1]+img[0]*mean_linear[0]
            elif c==2:
                mean=all_means[2]+img[0]*mean_linear[1]+img[1]*mean_linear[2]

            for i in range(remain_dim-1,-1,-1):
                cdf_min,p =dis_logistic_stats(mean[i],log_scales[c,i],int(img[c,i]*255),opt['p_prec'])
                ansstack.push(cdf_min,p)

    else:
        img=img.reshape(-1)
        if opt['obs_dis']=='Bernoulli':
            x_p=np.clip(sigmoid(x_p),1e-3,1-1e-3).reshape(-1)
            for i in range(opt['x_dim']-1,-1,-1):
                cdf_min,p =bernoulli_stats(x_p[i],int(img[i]),opt['p_prec'])
                ansstack.push(cdf_min,p)
        elif opt['obs_dis']=='Logistic':
            c=int(x_p.shape[1]//2)
            means,log_scales=x_p[:,:c,:,:].reshape(-1),x_p[:,c:,:,:].reshape(-1)

            for i in range(opt['x_dim']-1,-1,-1):
                cdf_min,p =dis_logistic_stats(means[i], log_scales[i],int(img[i]*255),opt['p_prec'])
                ansstack.push(cdf_min,p)
        else:
            raise NotImplementedError

    for i in range(opt['z_dim']-1,-1,-1):
        ansstack.push(index_list[i],1)

    return ansstack


def BBDecompression(ansstack,model,opt):
    sample_list=[]
    index_list=[]
    recover_img=[]

    model.encoder.eval()
    model.decoder.eval()
    encoder=model.encoder
    decoder=model.decoder

    for i in range(opt['z_dim']):
        x=ansstack.pop()
        ansstack.update(x,x,1)
        index_list.append(x)
        sample_list.append(opt['discretization_centres'][x])
    z_sample=np.asarray(sample_list)


    with torch.no_grad():
        x_p=decoder(torch.tensor(z_sample,dtype=torch.float32).to(opt['device'])).cpu().numpy()
    
    if opt['obs_dis']=='LogisticCA':
        c_num=3
        all_means=x_p[:,:c_num,:,:].reshape(c_num,-1)
        log_scales=x_p[:,c_num:2*c_num,:,:].reshape(c_num,-1)
        mean_linear=x_p[:,2*c_num:3*c_num,:,:].reshape(c_num,-1)

        recover_img=[]
        
        for c in range(0,c_num):

            if c==0:
                mean=all_means[0]
            elif c==1:
                mean=all_means[1]+recover_img[0]*mean_linear[0]
            else:
                mean=all_means[2]+recover_img[0]*mean_linear[1]+recover_img[1]*mean_linear[2]
                
            c_img=[]
            for i in range(int(opt['x_dim']//c_num)):
                s_bar=ansstack.pop()
                x,cdf_min,p=dis_logistic_ppf(mean[i],log_scales[c,i],s_bar,opt['p_prec'])
                ansstack.update(s_bar,cdf_min,p)
                c_img.append(x/255.)
            recover_img.append(np.asarray(c_img,np.float32))
            

    elif opt['obs_dis']=='Bernoulli':
        x_p=np.clip(sigmoid(x_p),1e-3,1-1e-3).reshape(-1)
        for i in range(opt['x_dim']):
            s_bar=ansstack.pop()
            x,cdf_min,p=bernoulli_ppf(x_p[i],s_bar,opt['p_prec'])
            ansstack.update(s_bar,cdf_min,p)
            recover_img.append(x)
    elif opt['obs_dis']=='Logistic':
        c=int(x_p.shape[1]//2)
        means,log_scales=x_p[:,:c,:,:].reshape(-1),x_p[:,c:,:,:].reshape(-1)
        for i in range(opt['x_dim']):
            s_bar=ansstack.pop()
            x,cdf_min,p=dis_logistic_ppf(means[i],log_scales[i],s_bar,opt['p_prec'])
            ansstack.update(s_bar,cdf_min,p)
            recover_img.append(x/255.)

    else:
        raise NotImplementedError

    recover_img=np.asarray(recover_img,np.float32)
    if opt['data_set']=='CIFAR':
        recover_img=recover_img.reshape(-1,3,32,32)
    if opt['optimal']==False:
        with torch.no_grad():
            mean,std=encoder(torch.tensor(recover_img).to(opt['device']))
        mean=mean.cpu().numpy()[0]
        std=std.cpu().numpy()[0]+1e-7
    else:
        mean,std=optimal_posterior(recover_img,model,opt)


    for i in range(opt['z_dim']-1,-1,-1):
        cdf_min,p = discrete_gaussian_stats(opt['discretization_scheme'],mean[i],std[i],index_list[i],opt['p_prec'])
        ansstack.push(cdf_min,p)

    return np.asarray(recover_img), ansstack




