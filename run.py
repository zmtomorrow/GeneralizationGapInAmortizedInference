import torch
from main import *
import os


opt = {}
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    opt["device"] = torch.device("cuda:0")
    opt["if_cuda"] = True
else:
    opt["device"] = torch.device("cpu")
    opt["if_cuda"] = False

opt['data_set']='CIFAR'
opt['x_dis']='Logistic' ## or MixLogistic 
opt['z_channels']=4
opt['z_dim']=opt['z_channels']*64

opt['epochs'] = 1000
opt['dataset_path']='../data/'
opt['save_path']='./'+opt['data_set']+'/'+opt['x_dis']+'_z'+str(opt['z_dim'])+'/'
if not os.path.exists(opt['save_path']):
    os.makedirs(opt['save_path'])
opt['result_path']=opt['save_path']
opt['batch_size'] = 100
opt['test_batch_size']=200
opt['if_regularizer']=False
opt['load_model']=False
opt['lr']=1e-4
opt['data_aug']=False
opt["seed"]=0
opt['if_save']=True
opt['save_epoch']=100
opt['additional_epochs']=100
opt['sample_size']=100
opt['if_save']=True

JointELBOTrain(opt)
