import torch
import torch.nn.functional as F


def discretized_logistic(x,l):
    means=l[:,0:3,:,:]
    log_scales=l[:,3:6,:,:]
    mean_linear=l[:,6:9,:,:]
    m2=means[:,1:2,:,:]+x[:,0:1,:,:]*mean_linear[:,0:1,:,:]
    m3=means[:,2:3,:,:]+x[:,0:1,:,:]*mean_linear[:,1:2,:,:]+mean_linear[:,2:3,:,:]*x[:,1:2,:,:]
    means=torch.cat((means[:,0:1,:,:],m2,m3),dim=1)                                                                           
    centered_x = x - means                                                                                                     
    inv_stdv = torch.exp(-torch.clamp(log_scales,min=-7.))                                                                     
    cdf_plus = torch.sigmoid(inv_stdv * (centered_x + 1. / 255.))
    cdf_min = torch.sigmoid(inv_stdv * centered_x )
    cdf_plus=torch.where(x > 0.999, torch.ones(1).to(x.device),cdf_plus)
    cdf_min=torch.where(x < 0.001, torch.zeros(1).to(x.device),cdf_min)
    return torch.sum(torch.log(cdf_plus-cdf_min+1e-7),(1,2,3))

def discretized_logistic_sample(l):
    means=l[:,0:3,:,:]
    log_scales=l[:,3:6,:,:]
    mean_linear=l[:,6:9,:,:]
    u=torch.rand_like(means)
    x=means + torch.exp(torch.clamp(log_scales,min=-7.)) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(x[:, 0:1, :, :], min=0., max=1.)
    x1 = torch.clamp(x[:, 1:2, :, :] + mean_linear[:, 0:1, :, :] * x0, min=0., max=1.)
    x2 = torch.clamp(x[:, 2:3, :, :] + mean_linear[:, 1:2, :, :] * x0 + mean_linear[:, 2:3, :, :] * x1, min=0., max=1.)
    return torch.cat((x0,x1,x2),1)
    
def discretized_mix_logistic_uniform(x, l, alpha=0.0001):
    xs=list(x.size())
    x=x.unsqueeze(2)
    mix_num = int(l.size(1)/10) 
    pi = torch.softmax(l[:, :mix_num,:,:],1).unsqueeze(1).repeat(1,3,1,1,1)
    l=l[:, mix_num:,:,:].view(xs[:2]+[-1]+xs[2:])
    means = l[:, :, :mix_num, :,:]
    inv_stdv = torch.exp(-torch.clamp(l[:, :, mix_num:2*mix_num,:, :], min=-7.))
    coeffs = torch.tanh(l[:, :, 2*mix_num:, : ,  : ])
    m2 = means[:,  1:2, :,:, :]+coeffs[:,  0:1, :,:, :]* x[:, 0:1, :,:, :]
    m3 = means[:,  2:3, :,:, :]+coeffs[:,  1:2, :,:, :] * x[:, 0:1,:,:, :]+coeffs[:,  2:3,:,:, :] * x[:,  1:2,:,:, :]
    means = torch.cat((means[:, 0:1,:, :, :],m2, m3), dim=1)
    centered_x = x - means
    cdf_plus = torch.sigmoid(inv_stdv * (centered_x + 1. / 510.))
    cdf_plus=torch.where(x > 0.9995, torch.tensor(1.0).to(x.device),cdf_plus)
    cdf_min = torch.sigmoid(inv_stdv * (centered_x - 1. / 510.))
    cdf_min=torch.where(x < 0.0005, torch.tensor(0.0).to(x.device),cdf_min)
    log_probs =torch.log((1-alpha)*(pi*(cdf_plus-cdf_min)).sum(2)+alpha*(1/256))
    return log_probs.sum([1,2,3])


def discretized_mix_logistic_sample(l):
    nr_mix= int(l.size(1)/10) 
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    temp = torch.clamp(torch.rand_like(logit_probs),1e-5, 1. - 1e-5).to(l.device)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)
    one_hot = F.one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4) 
    log_scales = torch.clamp(torch.sum(l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    coeffs = torch.sum(F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=4)
    u = torch.clamp(torch.rand_like(means),1e-5, 1. - 1e-5).to(l.device)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(x[:, :, :, 0], min=0., max=1.)
    x1 = torch.clamp(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=0., max=1.)
    x2 = torch.clamp(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=0., max=1.)
    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    out = out.permute(0, 3, 1, 2)
    return (out*255).int()/255.




