import time,os,sys

# import librosa, librosa.display 
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from network import Multimodal_Transformer, Signal_Encoder, Signal_Decoder, Frequency_2D_Encoder, AD_MODEL, weights_init, print_network

dirname=os.path.dirname
sys.path.insert(0,dirname(dirname(os.path.abspath(__file__))))

from metric import evaluate


class Discriminator(nn.Module):

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        
        #- signal
        model_s = Signal_Encoder(opt.ngpu,opt,1)
        layers_s = list(model_s.main.children())

        self.features = nn.Sequential(*layers_s[:-1])
        self.classifier = nn.Sequential(layers_s[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features


class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.bs = opt.batchsize
        self.signal_encoder = Signal_Encoder(opt.ngpu,opt,opt.nz)
        self.freq_2d_encoder = Frequency_2D_Encoder(opt.ngpu,opt,opt.nz)

        #self.tf = Multimodal_Transformer(ntoken=128, ninp=50, nhead=5, nhid=512, dropout=0.0, nlayers=3)
        self.tf = Multimodal_Transformer(bs=self.bs, ntoken=128, ninp=50, nhead=5, nhid=512, dropout=0.0, nlayers=3)
        
        self.signal_decoder = Signal_Decoder(opt.ngpu,opt)


    def forward(self, x_sig, x_freq):
        # Unimodal Encoder
        x_sig = self.signal_encoder(x_sig) # [bs,50,1]
        x_freq = self.freq_2d_encoder(x_freq).squeeze(3) #[bs,50,1]
        
        
        # Multimodal Transformer
        cls_token_s, cls_token_sf = self.tf(x_sig, x_freq)

        # Signal Decoder
        gen_signal = self.signal_decoder(cls_token_s) #unimodal decoder

        return gen_signal, cls_token_s, cls_token_sf



class BeatGAN(AD_MODEL):


    def __init__(self, opt, dataloader, device):
        super(BeatGAN, self).__init__(opt, dataloader, device)
        self.dataloader = dataloader
        self.device = device
        self.opt=opt

        self.batchsize = opt.batchsize
        self.nz = opt.nz
        self.niter = opt.niter

        self.G = Generator(opt).to(device)
        self.G.apply(weights_init)
        if not self.opt.istest:
            print_network(self.G)

        self.D = Discriminator(opt).to(device)
        self.D.apply(weights_init)
        if not self.opt.istest:
            print_network(self.D)


        self.bce_criterion = nn.BCELoss().cuda()
        self.mse_criterion=nn.MSELoss().cuda()


        self.optimizerD = optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


        self.total_steps = 0
        self.cur_epoch=0


        self.input_s = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)
        self.input_f = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt_s    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.gt_f    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input_s = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize),dtype=torch.float32, device=self.device)
        self.fixed_input_f = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize),dtype=torch.float32, device=self.device)
        self.real_label = 1
        self.fake_label= 0


        self.out_d_real = None
        self.feat_real = None

        self.fake = None
        self.latent_i = None
        self.out_d_fake = None
        self.feat_fake = None

        self.err_d_real = None
        self.err_d_fake = None
        self.err_d = None

        self.out_g = None
        self.err_g_adv = None
        self.err_g_rec = None
        self.err_g = None
        
        self.test_pair=[]

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['D_loss_real'] = []
        self.train_hist['D_loss_fake'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['G_loss_rec'] = []
        self.train_hist['G_loss_adv'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.train_hist['auc_s']=[]

        print("Train model.")
        start_time = time.time()
        best_auc_s = 0
        best_auc_epoch_s = 0
        best_auc_f = 0
        best_auc_epoch_f = 0
        
        results_path = os.path.join("results", "train")
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        with open(os.path.join(self.outf, self.model, self.dataset, "val_info.txt"), "w") as f:
            for epoch in range(self.niter):
                self.cur_epoch+=1

               #- train, validation
                self.train_epoch()
                auc_s,th_s,f1_s=self.validate()
                self.train_hist["auc_s"].append(auc_s)
                self.save(self.train_hist)
                self.save_loss(self.train_hist)
                self.save_auc(self.train_hist)

                if auc_s > best_auc_s: #val auc
                    best_auc_s = auc_s
                    best_auc_epoch_s=self.cur_epoch
                    self.save_weight_GD_S()

                print("[{}] auc_s:{:.4f} th_s:{:.4f} f1_s:{:.4f} \t best_auc:{:.4f} in epoch[{}]\n".format(self.cur_epoch,auc_s,th_s,f1_s,best_auc_s,best_auc_epoch_s))
                # print("[{}] auc_f:{:.4f} th_f:{:.4f} f1_f:{:.4f} \t best_auc:{:.4f} in epoch[{}]\n".format(self.cur_epoch,auc_f,th_f,f1_f,best_auc_f,best_auc_epoch_f))


        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.niter,
                                                                        self.train_hist['total_time'][0]))


    def train_epoch(self):

        epoch_start_time = time.time()
        self.G.train()
        self.D.train()
        epoch_iter = 0

        err_d, err_d_real, err_d_fake, err_g, err_g_rec, err_g_adv= 0.,0.,0.,0.,0.,0.
        num_batch = len(self.dataloader["train"])

        for data in self.dataloader["train"]:
            self.total_steps += self.opt.batchsize
            epoch_iter += 1

            self.set_input(data)
            self.optimize()

            errors = self.get_errors()
            err_d += errors["err_d"]
            err_d_real += errors["err_d_real"]
            err_d_fake += errors["err_d_fake"]
            
            err_g += errors["err_g"]
            err_g_rec += errors["err_g_rec"]
            err_g_adv += errors["err_g_adv"]


            if (epoch_iter  % self.opt.print_freq) == 0:
                print("Epoch: [%d] [%4d/%4d] D_loss(r/f): %.6f/%.6f, G_loss(adv/rec/dist): %.6f/%.6f/%.6f" %
                      ((self.cur_epoch), (epoch_iter), self.dataloader["train"].dataset.__len__() // self.batchsize,
                       errors["err_d_real"],errors["err_d_fake"],errors["err_g_adv"],errors["err_g_rec"],errors["err_g_distill"]))
                
                       
        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        self.train_hist['D_loss'].append(err_d/num_batch)
        self.train_hist['D_loss_real'].append(err_d_real/num_batch)
        self.train_hist['D_loss_fake'].append(err_d_fake/num_batch)
                
        self.train_hist['G_loss'].append(err_g/num_batch)
        self.train_hist['G_loss_rec'].append(err_g_rec/num_batch)
        self.train_hist['G_loss_adv'].append(err_g_adv/num_batch)
        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

        with torch.no_grad():
            # x_types = ['s','f']
            # for x_type in x_types:
            real_input,fake_output = self.get_generated_x()

            # self.visualize_pair_results(self.cur_epoch,
            #                             real_input,
            #                             fake_output,
            #                             is_train=True) 
            #                             #sample_type=x_type)


    def set_input(self, input):
        self.input_s.resize_(input[0][0].size()).copy_(input[0][0])
        self.gt_s.resize_(input[0][1].size()).copy_(input[0][1])

        self.input_f.resize_(input[1][0].size()).copy_(input[1][0])
        self.gt_f.resize_(input[1][1].size()).copy_(input[1][1])


        # fixed input for view
        if self.total_steps == self.opt.batchsize:
            self.fixed_input_s.resize_(input[0][0].size()).copy_(input[0][0])
            self.fixed_input_f.resize_(input[1][0].size()).copy_(input[1][0])


    def optimize(self):
        self.update_netd()
        self.update_netg()



    def update_netd(self):
        self.D.zero_grad()


        self.fake_s, _, _ = self.G(self.input_s, self.input_f)
        self.out_d_real_s, self.feat_real_s = self.D(self.input_s)
        self.out_d_fake_s, self.feat_fake_s = self.D(self.fake_s)

        self.err_d_real = self.bce_criterion(self.out_d_real_s, torch.full((self.batchsize,), self.real_label, device=self.device).type(torch.float32).cuda())
        self.err_d_fake = self.bce_criterion(self.out_d_fake_s, torch.full((self.batchsize,), self.fake_label, device=self.device).type(torch.float32).cuda())

        self.err_d = self.err_d_real + self.err_d_fake
        self.err_d.backward()
        self.optimizerD.step()


    def update_netg(self):
        self.G.zero_grad()

        self.fake_s, self.cls1, self.cls2 = self.G(self.input_s, self.input_f)
        self.out_d_real_s, self.feat_real_s = self.D(self.input_s)
        self.out_d_fake_s, self.feat_fake_s = self.D(self.fake_s)

        # feature matching loss
        self.err_g_adv = self.mse_criterion(self.feat_fake_s, self.feat_real_s)  # loss for feature matching
        self.err_distill = self.mse_criterion(self.cls1, self.cls2) # inputL cls_s, target: cls_sf

        # reconstruction loss
        self.err_g_rec = self.mse_criterion(self.fake_s, self.input_s)  # constrain x' to look like x

        self.err_g =  self.err_g_rec + self.err_g_adv + self.err_distill#* self.opt.w_adv
        self.err_g.backward()
        self.optimizerG.step()


    ##
    def get_errors(self):

        errors = {'err_d':self.err_d.item(),
                    'err_g': self.err_g.item(),
                    'err_d_real': self.err_d_real.item(),
                    'err_d_fake': self.err_d_fake.item(),
                    'err_g_adv': self.err_g_adv.item(),
                    'err_g_rec': self.err_g_rec.item(),
                    'err_g_distill': self.err_distill.item(),
                  }

        return errors



    def get_generated_x(self): #, x_type='s'):
        fake_s, _, _ = self.G(self.fixed_input_s, self.fixed_input_f)
        
        return  self.fixed_input_s.cpu().data.numpy(), fake_s.cpu().data.numpy()
            

    def train_test_type(self):
        self.G.eval()
        self.D.eval()
        res_th=self.opt.threshold
        save_dir = os.path.join(self.outf, self.model, self.dataset, "test", str(self.opt.folder))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        results_path = os.path.join("results", "train")
        if not os.path.exists(results_path):
            os.makedirs(results_path)
                
        y_N, y_pred_N = self.predict(self.dataloader["test_N"],scale=False)
        y_S, y_pred_S = self.predict(self.dataloader["test_S"],scale=False)
        y_V, y_pred_V = self.predict(self.dataloader["test_V"],scale=False)
        y_F, y_pred_F = self.predict(self.dataloader["test_F"],scale=False)
        y_Q, y_pred_Q = self.predict(self.dataloader["test_Q"],scale=False)
        over_all=np.concatenate([y_pred_N,y_pred_S,y_pred_V,y_pred_F,y_pred_Q])
        over_all_gt=np.concatenate([y_N,y_S,y_V,y_F,y_Q])
        min_score,max_score=np.min(over_all),np.max(over_all)
        A_res={
            "S":y_pred_S,
            "V":y_pred_V,
            "F":y_pred_F,
            "Q":y_pred_Q
        }
        self.analysisRes(y_pred_N,A_res,min_score,max_score,res_th,save_dir)

        aucprc,aucroc,best_th,best_f1=evaluate(over_all_gt,(over_all-min_score)/(max_score-min_score))
        print("#################################")
        print("########## Test Result ##########")
        print("ap:{}".format(aucprc))
        print("auc:{}".format(aucroc))
        print("best th:{} --> best f1:{}".format(best_th,best_f1))
        return aucroc


    def validate(self):
        '''
        validate by auc value
        :return: auc
        '''
        y_s, y_pred_s=self.predict(self.dataloader["val"], scale=True, is_train=True)
        rocprc_s,rocauc_s,best_th_s,best_f1_s=evaluate(y_s, y_pred_s)
        # rocprc_f,rocauc_f,best_th_f,best_f1_f=evaluate(y_f, y_pred_f)
        return rocauc_s,best_th_s,best_f1_s #, rocauc_f,best_th_f,best_f1_f


    def predict(self,dataloader_,scale=True, is_train=False):
        with torch.no_grad():

            self.an_scores_s = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels_s = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.long,    device=self.device)
 
            for i, data in enumerate(dataloader_, 0):

                self.set_input(data)
                self.fake_s, _, _ = self.G(self.input_s, self.input_f)

                error_s = torch.mean(
                    torch.pow((self.input_s.view(self.input_s.shape[0], -1) - self.fake_s.view(self.fake_s.shape[0], -1)), 2),
                    dim=1)
                self.an_scores_s[i*self.opt.batchsize : i*self.opt.batchsize+error_s.size(0)] = error_s.reshape(error_s.size(0))
                self.gt_labels_s[i*self.opt.batchsize : i*self.opt.batchsize+error_s.size(0)] = self.gt_s.reshape(error_s.size(0))

            # Scale error vector between [0, 1]
            if scale:
                self.an_scores_s = (self.an_scores_s - torch.min(self.an_scores_s)) / (torch.max(self.an_scores_s) - torch.min(self.an_scores_s))

            y_s=self.gt_labels_s.cpu().numpy()
            y_pred_s=self.an_scores_s.cpu().numpy()

            return y_s, y_pred_s

   
    def predict_for_right(self,dataloader_,min_score,max_score,threshold,save_dir,data_type='s'):
        '''

        :param dataloader:
        :param min_score:
        :param max_score:
        :param threshold:
        :param save_dir:
        :return:
        '''
        assert  save_dir is not None
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.G.eval()
        self.D.eval()
        with torch.no_grad():
            # Create big error tensor for the test set.
            test_pair2=[]
           
            self.an_scores_s = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)

            for i, data in enumerate(dataloader_, 0):
                self.set_input(data)
                self.fake_s, _, _ = self.G(self.input_s, self.input_f)

                error_s = torch.mean(
                    torch.pow((self.input_s.view(self.input_s.shape[0], -1) - self.fake_s.view(self.fake_s.shape[0], -1)), 2),
                    dim=1)
                # self.an_scores_s[i*self.opt.batchsize : i*self.opt.batchsize+error_s.size(0)] = error_s.reshape(error_s.size(0))

                '''
                error_f = torch.mean(
                    torch.pow((self.input_f.view(self.input_f.shape[0], -1) - self.fake_f.view(self.fake_f.shape[0], -1)), 2),
                    dim=1)
                '''    
                #print('[predict_for_right] fake_f', self.fake_f.shape, 'input_s', self.input_s.shape)
                error_f = torch.mean(
                    torch.pow((self.input_s.view(self.input_s.shape[0], -1) - self.fake_f.view(self.fake_f.shape[0], -1)), 2),
                    dim=1)
                # self.an_scores_f[i*self.opt.batchsize : i*self.opt.batchsize+error_f.size(0)] = error_f.reshape(error_f.size(0))

                
                if i==0 and data_type=='f':
                     real_s = self.input_s.cpu().numpy()
                     real_f = self.input_f.cpu().numpy()
                     fake_f = self.fake_f.cpu().numpy()
                     
                    #  #--save real signal/real freq
                    #  fig = plt.figure(figsize=(5,10))
                    #  plt.subplot(2,1,1)
                    #  plt.plot(real_s[0][0])
                     
                    #  plt.subplot(2,1,2)
                    #  img = librosa.display.specshow(real_f[0][0], sr=360, hop_length = 2, y_axis="linear", x_axis="time")
                    
                    #  fig.savefig("results/test/real_{0}.png".format(save_dir[-1]))
                    #  #--
                     
                    #  #--save real signal/fake siganl
                    #  fig = plt.figure(figsize=(5,10))
                    #  plt.subplot(2,1,1)
                    #  plt.plot(real_s[0][0])
                    
                    #  plt.subplot(2,1,2)
                    #  plt.plot(fake_f[0][0])
                    #  fig.savefig("results/test/fake_{0}.png".format(save_dir[-1]))
                    #  #--
                     
                     self.test_pair.append((real_s[0],fake_f[0]))
                
                
                # Save test images.
                batch_input = None
                batch_output = None
                ano_score = None
                
                if data_type == 's':
                    batch_input = self.input_s.cpu().numpy()
                    batch_output = self.fake_s.cpu().numpy()
                    ano_score=error_s.cpu().numpy()
                    
                else : #data_type == 'f'
                    batch_input = self.input_s.cpu().numpy()
                    batch_output = self.fake_f.cpu().numpy()
                    ano_score=error_f.cpu().numpy()
                    
                assert batch_output.shape[0]==batch_input.shape[0]==ano_score.shape[0]
                for idx in range(batch_input.shape[0]):
                    if len(test_pair2)>=100: 
                        break
                    normal_score=(ano_score[idx]-min_score)/(max_score-min_score)

                    if normal_score>=threshold:
                        test_pair2.append((batch_input[idx],batch_output[idx]))
            
        if data_type == 'f':
            self.saveTestPair(self.test_pair,'results/test')
        self.saveTestPair(test_pair2,save_dir)


    def test_type_signal(self):
        self.G.eval()
        self.D.eval()
        res_th=self.opt.threshold
        save_dir = os.path.join(self.outf, self.model, self.dataset, "test", str(self.opt.folder), "sig")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        y_N, y_pred_N = self.predict(self.dataloader["test_N"],scale=False)
        y_S, y_pred_S = self.predict(self.dataloader["test_S"],scale=False)
        y_V, y_pred_V = self.predict(self.dataloader["test_V"],scale=False)
        y_F, y_pred_F = self.predict(self.dataloader["test_F"],scale=False)
        y_Q, y_pred_Q = self.predict(self.dataloader["test_Q"],scale=False)

        over_all=np.concatenate([y_pred_N,y_pred_S,y_pred_V,y_pred_F,y_pred_Q])
        over_all_gt=np.concatenate([y_N,y_S,y_V,y_F,y_Q])
        min_score,max_score=np.min(over_all),np.max(over_all)
        A_res={
            "S":y_pred_S,
            "V":y_pred_V,
            "F":y_pred_F,
            "Q":y_pred_Q
        }
        self.analysisRes(y_pred_N,A_res,min_score,max_score,res_th,save_dir)


        #save fig for Interpretable
        # self.predict_for_right(self.dataloader["test_N"], min_score,max_score,res_th,save_dir=os.path.join(save_dir, "N"),data_type='s')
        # self.predict_for_right(self.dataloader["test_S"], min_score,max_score,res_th,save_dir=os.path.join(save_dir, "S"),data_type='s')
        # self.predict_for_right(self.dataloader["test_V"], min_score,max_score,res_th,save_dir=os.path.join(save_dir, "V"),data_type='s')
        # self.predict_for_right(self.dataloader["test_F"], min_score,max_score,res_th,save_dir=os.path.join(save_dir, "F"),data_type='s')
        # self.predict_for_right(self.dataloader["test_Q"], min_score,max_score,res_th,save_dir=os.path.join(save_dir, "Q"),data_type='s')
        aucprc,aucroc,best_th,best_f1=evaluate(over_all_gt,(over_all-min_score)/(max_score-min_score))
        print("#############################")
        print("########  Result  ###########")
        print("ap:{}".format(aucprc))
        print("auc:{}".format(aucroc))
        print("best th:{} --> best f1:{}".format(best_th,best_f1))


        with open(os.path.join(save_dir,"res-record.txt"),'w') as f:
            f.write("auc_prc:{}\n".format(aucprc))
            f.write("auc_roc:{}\n".format(aucroc))
            f.write("best th:{} --> best f1:{}".format(best_th, best_f1))



    def test_time(self):
        self.G.eval()
        self.D.eval()
        size=self.dataloader["test_N"].dataset.__len__()
        start=time.time()

        for i, (data_x,data_y) in enumerate(self.dataloader["test_N"], 0):
            input_x=data_x
            for j in range(input_x.shape[0]):
                input_x_=input_x[j].view(1,input_x.shape[1],input_x.shape[2]).to(self.device)
                digit, _, gen_x = self.G(input_x_)

                error = torch.mean(
                    torch.pow((digit.view(digit.shape[0], -1) - gen_x.view(gen_x.shape[0], -1)), 2),
                    dim=1)

        end=time.time()
        print((end-start)/size)
