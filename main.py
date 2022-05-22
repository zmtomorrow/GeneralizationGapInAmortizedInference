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
            eval_BPD+=-model(x.to(opt['device'])).item()
        return eval_BPD/(len(dataloader)*np.prod(x.size()[-3:]))

def JointELBOTrain(opt):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    train_data,test_data,train_data_evaluation=LoadData(opt)

    model=VAE(opt).to(opt['device'])

    optimizer = optim.Adam(model.parameters(), lr=opt['lr'])

    train_BPD_list=[]
    test_BPD_list=[]
    for epoch in range(1, opt['epochs'] + 1):
        model.train()
        for x, _ in tqdm(train_data):
            optimizer.zero_grad()
            L = -model(x.to(opt['device']))
            L.backward()
            optimizer.step()

        train_BPD=Eval(model,train_data_evaluation,opt)
        test_BPD=Eval(model,test_data,opt)
        print('epoch:',epoch,train_BPD,test_BPD)
        train_BPD_list.append(train_BPD)
        test_BPD_list.append(test_BPD)
        np.save(opt['result_path']+'testBPD',test_BPD_list)
        np.save(opt['result_path']+'trainBPD',train_BPD_list)
        
        if epoch%opt['save_epoch']==0:
            torch.save(model.state_dict(),opt['save_path']+'vae_epoch'+str(epoch)+'.pth')

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
        torch.save(model.state_dict(),opt['save_path']+'ws_epoch'+str(epoch)+'.pth')
    return train_BPD_list,test_BPD_list


def Sleep(opt,load_name):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    train_data,test_data,train_data_evaluation=LoadData(opt)

    model=VAE(opt).to(opt['device'])
    model.load_state_dict(torch.load(opt['save_path']+load_name))
    optimizer = optim.Adam(model.encoder.parameters(), lr=opt['lr'])
#     print('start',Eval(model,test_data,opt))
    test_BPD_list=[]
    test_BPD_list.append(Eval(model,test_data,opt))

    for epoch in tqdm(range(1, opt['additional_epochs'] + 1)):
        model.decoder.eval()
        model.encoder.train()
        for x,_ in train_data:
            optimizer.zero_grad()
            with torch.no_grad():
                z_samples,samples= model.sample()
            z_mu, z_std = model.encoder(samples.to(opt['device']))
            logqz_x=Normal(z_mu,z_std).log_prob(z_samples.to(opt['device'])).sum(-1)
            L = -logqz_x.mean(0)
            L.backward()
            optimizer.step()
        
        test_BPD=Eval(model,test_data,opt)
        test_BPD_list.append(test_BPD)
#         print('epoch:',epoch,test_BPD)
    if opt['if_save']:
        torch.save(model.state_dict(),opt['save_path']+load_name.split('.')[0]+'_sleep.pth')
    return test_BPD_list


def Wake(opt,load_name):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    train_data,test_data,train_data_evaluation=LoadData(opt)

    model=VAE(opt).to(opt['device'])
    model.load_state_dict(torch.load(opt['save_path']+load_name))
    optimizer = optim.Adam(model.encoder.parameters(), lr=opt['lr'])
#     print('start',Eval(model,test_data,opt))
    test_BPD_list=[]
    test_BPD_list.append(Eval(model,test_data,opt))

    for epoch in tqdm(range(1, opt['additional_epochs'] + 1)):
        model.decoder.eval()
        model.encoder.train()
        for x,_ in train_data:
            optimizer.zero_grad()
            L = -model(x.to(opt['device']))
            L.backward()
            optimizer.step()
        
        test_BPD=Eval(model,test_data,opt)
        test_BPD_list.append(test_BPD)
#         print('epoch:',epoch,test_BPD)
    if opt['if_save']:
        torch.save(model.state_dict(),opt['save_path']+load_name.split('.')[0]+'_wake.pth')
    return test_BPD_list

def ReverseSleep(opt,load_name):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    train_data,test_data,train_data_evaluation=LoadData(opt)

    model=VAE(opt).to(opt['device'])
    model.load_state_dict(torch.load(opt['save_path']+load_name))
    optimizer = optim.Adam(model.encoder.parameters(), lr=opt['lr'])
#     print('start',Eval(model,test_data,opt))
    test_BPD_list=[]
    test_BPD_list.append(Eval(model,test_data,opt))
    for epoch in tqdm(range(1, opt['additional_epochs'] + 1)):
        model.decoder.eval()
        model.encoder.train()
        for _ in train_data:
            _,data_sample=model.sample(opt['sample_size'])
            optimizer.zero_grad()
            L = -model(data_sample)
            L.backward()
            optimizer.step()
        
        test_BPD=Eval(model,test_data,opt)
#         print('epoch:',epoch,test_BPD)
        test_BPD_list.append(test_BPD)
    if opt['if_save']:
        torch.save(model.state_dict(),opt['save_path']+load_name.split('.')[0]+'_revsleep.pth')
    return test_BPD_list

def ReverseHalfAsleep(opt,load_name):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    save_name=opt['save_path']+load_name.split('.')[0]+'_improved'

    train_data,test_data,train_data_evaluation=LoadData(opt)

    model=VAE(opt).to(opt['device'])
    model.load_state_dict(torch.load(opt['save_path']+load_name))
    optimizer = optim.Adam(model.encoder.parameters(), lr=opt['lr'])
    test_BPD=Eval(model,test_data,opt)
    
#     file_write=open(save_name,'a')
#     file_write.write('start test_BPD:'+str(test_BPD)+'\n')
    
    test_BPD_list=[]
    test_BPD_list.append(test_BPD)

    for epoch in tqdm(range(1, opt['additional_epochs'] + 1)):
        model.decoder.eval()
        model.encoder.train()

        loss_list=[]
        for x, _ in train_data:
            with torch.no_grad():
                if opt['sample_qz']:
                    model.encoder.eval()            
                    z_mu, z_std = model.encoder(x.to(opt['device']))
                    eps = torch.randn_like(z_mu).to(opt['device'])
                    zs = eps.mul(z_std).add_(z_mu)
                    pxz_params = model.decoder(zs)
                    data_sample=model.sample_op(pxz_params)
                    model.encoder.train()
                else:
                    data_sample=model.sample(opt['sample_size'])
            x=torch.cat((x.to(opt['device']),data_sample),0)
            optimizer.zero_grad()

            L = -model(x)
            L.backward()
            optimizer.step()

        test_BPD=Eval(model,test_data,opt)
        test_BPD_list.append(test_BPD)
        
    if opt['if_save']:
        torch.save(model.state_dict(),save_name+'.pth')
        np.save(save_name,test_BPD_list)

    return test_BPD_list


def Denoising(opt,load_name):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    train_data,test_data,train_data_evaluation=LoadData(opt)

    model=VAE(opt).to(opt['device'])
    model.load_state_dict(torch.load(opt['save_path']+load_name))
    optimizer = optim.Adam(model.encoder.parameters(), lr=opt['lr'])
    print('start',Eval(model,test_data,opt))
    test_BPD_list=[]
    test_BPD_list.append(Eval(model,test_data,opt))

    for epoch in tqdm(range(1, opt['additional_epochs'] + 1)):
        model.decoder.eval()
        model.encoder.train()
        for x, _ in train_data:
            data_sample=x+torch.randn_like(x)*opt['std']
            tilde_x=torch.cat((x,data_sample),0).to(opt['device'])
            x=torch.cat((x,x),0).to(opt['device'])
            optimizer.zero_grad()
            L = -model.denoising_forward(tilde_x,x)
            L.backward()
            optimizer.step()
        
        test_BPD=Eval(model,test_data,opt)
        test_BPD_list.append(test_BPD)
        # print('epoch:',epoch,test_BPD)
    return test_BPD_list



def OptimalEval(opt,load_name):
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    
    train_data,test_data,train_data_evaluation=LoadData(opt)

    model=VAE(opt).to(opt['device'])
    model.load_state_dict(torch.load(opt['save_path']+load_name))
    optimizer = optim.Adam(model.encoder.parameters(), lr=opt['lr'])
    print('start',Eval(model,test_data,opt))
    test_list=[]
    for epoch in tqdm(range(1, opt['additional_epochs'] + 1)):
        model.decoder.eval()
        model.encoder.train()

        for x, _ in test_data:
            optimizer.zero_grad()
            L = -model(x.to(opt['device']))
            L.backward()
            optimizer.step()

        test_BPD=Eval(model,test_data,opt)
        test_list.append(test_BPD)
    return test_list
