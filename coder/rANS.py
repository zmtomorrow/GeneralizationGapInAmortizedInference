from scipy.stats import norm
import numpy as np
# from tools import *
# from utils import *
from scipy.special import expit as sigmoid
import torch.nn as nn

rng = np.random.RandomState(0)


class ANSStack(object):
    def __init__(self, s_prec, t_prec, p_prec, randominit=False):
        self.s_prec=int(s_prec)
        self.t_prec=int(t_prec)
        self.p_prec=int(p_prec)
        self.t_mask = (1 << t_prec) - 1
        self.s_min=1 << s_prec - t_prec
        self.s_max=1 << s_prec
            
        if randominit:
            def unflatten(arr,t_prec):
                return int(arr[0]) << t_prec | int(arr[1]), list(arr[2:])
            other_bits = rng.randint(low=1 << 16, high=1 << 31, size=200, dtype=np.uint32)
            s_init,t_init=unflatten(other_bits,32)
            self.s, self.t_stack=int(s_init),[int(i) for i in t_init]
            self.init_length=self.get_length()
        else:
            self.s, self.t_stack= self.s_min, [] 
            self.init_length=self.get_length()

    def push(self,c_min,p):
        c_min,p=int(c_min),int(p)
        while self.s >= p << (self.s_prec - self.p_prec):
            self.t_stack.append(self.s & self.t_mask )
            self.s=self.s>> self.t_prec
        self.s = (self.s//p << self.p_prec) + self.s%p + c_min
        assert self.s_min <= self.s < self.s_max

    def pop(self):
        return self.s & ((1 << self.p_prec) - 1)

    def update(self,s_bar,c_min,p):
        s_bar,c_min,p=int(s_bar),int(c_min),int(p)
        self.s = p * (self.s >> self.p_prec) + s_bar - c_min
        while self.s < self.s_min:
            t_top=self.t_stack.pop()
            self.s = (self.s << self.t_prec) + t_top
        assert self.s_min <= self.s < self.s_max
        
    def get_length(self):
        return len(self.t_stack)*self.t_prec+len(bin(self.s))



def _nearest_int(arr):
    return np.uint64(np.ceil(arr - 0.5))

def uniform_stats(idx):
    return idx,1

def std_gaussian_buckets(log_num):
    return np.float64(norm.ppf(np.arange((1 << log_num) + 1) / (1 << log_num)))

def std_gaussian_centres(log_num):
    return np.float32(norm.ppf((np.arange((1 << log_num)) + 0.5) / (1 << log_num)))


def discrete_gaussian_stats(buckets, mean, std, idx, prec):
    cdf_min=_nearest_int(norm.cdf(buckets[idx], mean, std) * (1 << prec))
    p=_nearest_int(norm.cdf(buckets[idx+1], mean, std) * (1 << prec))-cdf_min
    return  cdf_min, p

def discrete_gaussian_ppf(buckets,mean, std, s_bar, prec):
    x = norm.ppf((s_bar + 0.5) / (1 << prec), mean, std)
    return np.searchsorted(buckets, x, 'right') - 1

def bernoulli_stats(p, idx, prec):
    p_0=_nearest_int((1-p) * (1 << prec))
    if idx==0:
        return 0, int(p_0)
    else:
        return int(p_0), int((1 << prec)-p_0)

def bernoulli_ppf(p, s_bar, prec):
    p_0=_nearest_int((1-p) * (1 << prec))
    if s_bar < p_0:
        return 0, 0, int(p_0)
    else:
        return 1, int(p_0), int((1 << prec)-p_0)



def dis_logistic_stats_table(means, log_scales,p_prec):
    x_t=np.arange(0,256)
    centered_x =  x_t/255.- means   
    inv_stdv = np.exp(-np.clip(log_scales,a_min=-7.,a_max=10.))                                         
    cdf_plus = sigmoid(inv_stdv * (centered_x + 1. / 255.))    
    cdf_min = sigmoid(inv_stdv * centered_x)
    p_total=1<<p_prec
    cdf_min=_nearest_int(cdf_min* p_total)
    cdf_plus=_nearest_int(cdf_plus* p_total)
    cdf_min[0]=0
    cdf_plus[-1]=p_total
    probs=cdf_plus-cdf_min
    probs[probs==0]=1
    probs[np.argmax(probs)]+=(p_total-np.sum(probs))
    cdf_min=np.concatenate(([0],np.cumsum(probs)[:-1]))
    return cdf_min,probs


def dis_logistic_stats(means, log_scales, idx, prec):
    cdf_min_table,prob_table=dis_logistic_stats_table(means, log_scales,prec)
    return int(cdf_min_table[idx]),int(prob_table[idx])

def dis_logistic_ppf(means, log_scales, s_bar, prec):
    cdf_min_table,prob_table=dis_logistic_stats_table(means, log_scales,prec)
    x=np.searchsorted(cdf_min_table, s_bar, side='right', sorter=None)-1
    return x,int(cdf_min_table[x]),int(prob_table[x])



