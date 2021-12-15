import torch
import numpy as np
from torch.autograd import Variable 

def batch_KL_diag_gaussian_std(mu_1, std_1, mu_2, std_2):
    diag_1 = std_1 ** 2
    diag_2 = std_2 ** 2
    ratio = diag_1 / diag_2
    return 0.5 * (
        torch.sum((mu_1 - mu_2) ** 2 / diag_2, dim=-1)
        + torch.sum(ratio, dim=-1)
        - torch.sum(torch.log(ratio), dim=-1)
        - mu_1.size(1)
    )

def log_prob_from_logits(x):
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def to_one_hot(tensor, n, fill_with=1.):
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda : one_hot = one_hot.to(tensor.device)
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return Variable(one_hot)


def one_hot(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]

def vec_to_tril_matrix(vec, diag=0):
    n = (-(1 + 2 * diag) + ((1 + 2 * diag)**2 + 8 * vec.shape[-1] + 4 * abs(diag) * (diag + 1))**0.5) / 2
    n = torch.round(n).long() if isinstance(n, torch.Tensor) else round(n)
    mat = vec.new_zeros(vec.shape[:-1] + torch.Size((n, n)))
    arange = torch.arange(n, device=vec.device)
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    mat[..., tril_mask] = vec
    return mat


def off_diag_matrix(vec):
    l=int(np.sqrt(vec.size(-1)*2+0.25)+0.5)
    low_t_matrix=vec_to_tril_matrix(vec)
    pad1=torch.zeros(vec.shape[:-1]+torch.Size((l-1,1))).to(vec.device)
    pad2=torch.zeros(vec.shape[:-1]+torch.Size((1,l))).to(vec.device)
    return torch.cat((pad2,torch.cat((low_t_matrix,pad1),-1)),-2)
