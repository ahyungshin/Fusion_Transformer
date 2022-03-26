

import os
import torch
from options import Options
import numpy as np
from data import load_data
import random
import torch.backends.cudnn as cudnn

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
cudnn.benchmark = False
cudnn.deterministic = True
np.random.seed(random_seed)
random.seed(random_seed)

# from dcgan import DCGAN as myModel


device = torch.device("cuda:0" if
torch.cuda.is_available() else "cpu")
print('device: ',device)

opt = Options().parse()
print(opt)
dataloader=load_data(opt)
print("load data success!!!")

if opt.model == "beatgan":
    from model import BeatGAN as MyModel

else:
    raise Exception("no this model :{}".format(opt.model))


model=MyModel(opt,dataloader,device)
print('\nmodel_device:',model.device,'\n')

if not opt.istest:
    print("################  Train  ##################")
    model.train()
else:
    print("################  Eval  ##################")
    model.load()
    model.test_type_signal()
    # model.test_type_freq()
    # model.test_time()
    # model.plotTestFig()
    # print("threshold:{}\tf1-score:{}\tauc:{}".format( th, f1, auc))
