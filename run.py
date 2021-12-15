import torch
from train import *


opt = {}

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    opt["device"] = torch.device("cuda:0")
    opt["if_cuda"] = True
else:
    opt["device"] = torch.device("cpu")
    opt["if_cuda"] = False

#opt['data_set']='MNIST'
#opt['x_dis']='Logistic'

opt['data_set']='BinaryMNIST'
opt['x_dis']='Bernoulli'

opt['dataset_path']='../data/'
opt['save_path']='./save/'
opt['result_path']='./results/'
opt['epochs'] = 200
opt['batch_size'] = 100
opt['test_batch_size']=1000
opt['if_regularizer']=False
opt['load_model']=False
opt['lr']=1e-3
opt['z_dim']=10
opt['SelfConsistPhase']=False
opt['if_bn']=True
opt['h_layer_num']=2


JointELBO_train=[]
JointELBO_test=[]
PartialEM_train=[]
PartialEM_test=[]

for opt["seed"] in range(0,5):
    train_BPD, test_BPD = JointELBOTrain(opt)
    JointELBO_train.append(train_BPD)
    JointELBO_test.append(test_BPD)

    np.save(opt['result_path']+'JointELBO_train', JointELBO_train)
    np.save(opt['result_path']+'JointELBO_test', JointELBO_test)
    


