import torch
from torch import optim
from utils import *
from model import *
import numpy as np
from tqdm import tqdm
from torch.distributions.normal import Normal


def LoadEval(opt,load_name):
    train_data,test_data,train_data_evaluation=LoadData(opt)
    
    model=VAE(opt).to(opt['device'])
    model.load_state_dict(torch.load(opt['save_path']+load_name))

    train_BPD=Eval(model,train_data_evaluation,opt)
    test_BPD=Eval(model,test_data,opt)
    return train_BPD,test_BPD


def Eval(model,dataloader,opt):
    with torch.no_grad():
        model.eval()
        eval_BPD=0.
        for x, _ in dataloader:
            if opt['x_dis']=='MixLogistic':
                x = rescaling(x)
            else:
                pass
            eval_BPD+=-model(x.to(opt['device'])).item()
        return eval_BPD/(len(dataloader)*np.prod(x.size()[-3:]))

def JointELBOTrain(opt):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    train_data,test_data,train_data_evaluation=LoadData(opt)

    model=VAE(opt).to(opt['device'])
    if opt['load_model']==True:
        model.load_state_dict(torch.load(opt['save_path']+opt['load_name']))

    optimizer = optim.Adam(model.parameters(), lr=opt['lr'])

    train_BPD_list=[]
    test_BPD_list=[]
    for epoch in tqdm(range(1, opt['epochs'] + 1)):
        model.train()
        for x, _ in train_data:
            if opt['x_dis']=='MixLogistic':
                x = rescaling(x)
            else:
                pass
            optimizer.zero_grad()
            L = -model(x.to(opt['device']))
            L.backward()
            optimizer.step()


        train_BPD=Eval(model,train_data_evaluation,opt)
        test_BPD=Eval(model,test_data,opt)
        train_BPD_list.append(train_BPD)
        test_BPD_list.append(test_BPD)
        print('epoch:',epoch,train_BPD,test_BPD)
        if epoch%opt['save_epoch']==0:
            if opt['if_save']:
                torch.save(model.state_dict(),opt['save_path']+opt['data_set']+opt['x_dis']+'_epoch'+str(epoch)+'_joint.pth')
    return train_BPD_list,test_BPD_list



def WSTrain(opt):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    train_data,test_data,train_data_evaluation=LoadData(opt)

    model=VAE(opt).to(opt['device'])
    if opt['load_model']==True:
        model.load_state_dict(torch.load(opt['save_path']+opt['load_name']))

    enc_optimizer = optim.Adam(model.encoder.parameters(), lr=opt['lr'])
    dec_optimizer = optim.Adam(model.decoder.parameters(), lr=opt['lr'])

    train_BPD_list=[]
    test_BPD_list=[]
    for epoch in tqdm(range(1, opt['epochs'] + 1)):
        model.train()
        for x, _ in train_data:
            if opt['x_dis']=='MixLogistic':
                x = rescaling(x)
            else:
                pass
            #wake
            dec_optimizer.zero_grad()
            L = -model(x.to(opt['device']))
            L.backward()
            dec_optimizer.step()
            #sleep
            with torch.no_grad():
                z_samples,samples= model.sample()
            z_mu, z_std = model.encoder(samples.to(opt['device']))
            logqz_x=Normal(z_mu,z_std).log_prob(z_samples.to(opt['device'])).sum(-1)
            L = -logqz_x.mean(0)
            L.backward()
            enc_optimizer.step()

        train_BPD=Eval(model,train_data_evaluation,opt)
        test_BPD=Eval(model,test_data,opt)
        train_BPD_list.append(train_BPD)
        test_BPD_list.append(test_BPD)
        # print('epoch:',epoch,train_BPD,test_BPD)
        
    if opt['if_save']:
        torch.save(model.state_dict(),opt['save_path']+opt['data_set']+opt['x_dis']+'_epoch'+str(epoch)+'.pth')
    return train_BPD_list,test_BPD_list





def ReverseHalfAsleep(opt,load_name):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    train_data,test_data,train_data_evaluation=LoadData(opt)

    model=VAE(opt).to(opt['device'])
    model.load_state_dict(torch.load(opt['save_path']+load_name))
    optimizer = optim.Adam(model.encoder.parameters(), lr=opt['lr'])
    print('start',Eval(model,test_data,opt))
        
    for epoch in tqdm(range(1, opt['additional_epochs'] + 1)):
        model.decoder.eval()
        model.encoder.train()
        for x, _ in train_data:
            if opt['x_dis']=='MixLogistic':
                x = rescaling(x)
            else:
                pass
            _,data_sample=model.sample(opt['sample_size'])
            x=torch.cat((x.to(opt['device']),data_sample),0)
            optimizer.zero_grad()
            L = -model(x)
            L.backward()
            optimizer.step()
        
        test_BPD=Eval(model,test_data,opt)
        print('epoch:',epoch,test_BPD)
    return test_BPD





def OptimalEval(opt,load_name):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    
    train_data,test_data,train_data_evaluation=LoadData(opt)

    model=VAE(opt).to(opt['device'])
    model.load_state_dict(torch.load(opt['save_path']+load_name))
    optimizer = optim.Adam(model.encoder.parameters(), lr=opt['lr'])
    print('start',Eval(model,test_data,opt))
    for epoch in tqdm(range(1, opt['additional_epochs'] + 1)):
        model.decoder.eval()
        model.encoder.train()

        for x, _ in test_data:
            if opt['x_dis']=='MixLogistic':
                x = rescaling(x)
            else:
                pass
            optimizer.zero_grad()
            L = -model(x.to(opt['device']))
            L.backward()
            optimizer.step()

        test_BPD=Eval(model,test_data,opt)
        print(test_BPD)
    return test_BPD
