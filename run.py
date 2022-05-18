import torch
from torch import optim
from utils import *
from model import *
import numpy as np
from tqdm import tqdm
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
opt['z_channels']=16
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
opt['save_epoch']=50
opt['additional_epochs']=100
opt['sample_size']=100
opt['if_save']=True


np.random.seed(opt['seed'])
torch.manual_seed(opt['seed'])

train_data,test_data,train_data_evaluation=LoadData(opt)
model=VAE(opt).to(opt['device'])

# if opt['load_model']==True:
#     model.load_state_dict(torch.load(opt['save_path']+opt['load_name']))

optimizer = optim.Adam(model.parameters(), lr=opt['lr'])


test_BPD_list=[]
train_BPD_list=[]

for epoch in range(1, opt['epochs'] + 1):
    model.train()
    for x, _ in tqdm(train_data):
        optimizer.zero_grad()
        L = -model(x.to(opt['device']))
        L.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_BPD=0.
        for x, _ in train_data_evaluation:
            train_BPD+=-model(x.to(opt['device'])).item()
        train_BPD = train_BPD/(len(train_data_evaluation)*np.prod(x.size()[-3:]))
        print('epoch:',epoch,'train_BPD:',train_BPD)


        test_BPD=0.
        for x, _ in test_data:
            test_BPD+=-model(x.to(opt['device'])).item()
        test_BPD = test_BPD/(len(test_data)*np.prod(x.size()[-3:]))
        print('epoch:',epoch,'test_BPD:',test_BPD)
        
        
     
        test_BPD_list.append(test_BPD)
        train_BPD_list.append(train_BPD)

        np.save(opt['result_path']+'testBPD',test_BPD_list)
        np.save(opt['result_path']+'trainlBPD',train_BPD_list)
        
        if epoch%100==0:
            torch.save(model.state_dict(),opt['save_path']+'epoch'+str(epoch)+'.pth')
