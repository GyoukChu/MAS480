import torch
from torch.autograd import grad
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
torch.manual_seed(777) 
device = torch.device('cuda:1')

class PINN(nn.Module):
    def __init__(self, t, S, I, R, V, D):
        super(PINN, self).__init__()
        
        self.N = 5174e+4
        self.t = t.clone().detach().requires_grad_(True).to(device)
        self.t_float = self.t.float()
        self.t_batch = torch.reshape(self.t_float, (len(self.t), 1))

        self.S = S.to(device); self.I = I.to(device); self.R = R.to(device); 
        self.V = V.to(device); self.D = D.to(device)

        self.losses = []

        self.alpha_tilda = torch.nn.Parameter(torch.rand(1, device=device, requires_grad=True))
        self.beta_tilda = torch.nn.Parameter(torch.rand(1, device=device, requires_grad=True))
        self.gamma_tilda = torch.nn.Parameter(torch.rand(1, device=device, requires_grad=True))
        self.delta_tilda = torch.nn.Parameter(torch.rand(1, device=device, requires_grad=True))
        self.sigma_tilda = torch.nn.Parameter(torch.rand(1, device=device, requires_grad=True))

        # Do normalization : Drastic variation in large scale is hard to learn
        self.S_max = max(self.S); self.S_min = min(self.S)
        self.I_max = max(self.I); self.I_min = min(self.I)
        self.R_max = max(self.R); self.R_min = min(self.R)
        self.V_max = max(self.V); self.V_min = min(self.V)
        self.D_max = max(self.D); self.D_min = min(self.D)

        self.S_hat = (self.S - self.S_min) / (self.S_max - self.S_min)
        self.I_hat = (self.I - self.I_min) / (self.I_max - self.I_min)
        self.R_hat = (self.R - self.R_min) / (self.R_max - self.R_min)
        self.V_hat = (self.V - self.V_min) / (self.V_max - self.V_min)
        self.D_hat = (self.D - self.D_min) / (self.D_max - self.D_min)

        # matrices (x5 for S,I,R,V,D) for the gradients
        self.m1 = torch.zeros((len(self.t), 5), device=device); self.m1[:, 0] = 1
        self.m2 = torch.zeros((len(self.t), 5), device=device); self.m2[:, 1] = 1
        self.m3 = torch.zeros((len(self.t), 5), device=device); self.m3[:, 2] = 1
        self.m4 = torch.zeros((len(self.t), 5), device=device); self.m4[:, 3] = 1
        self.m5 = torch.zeros((len(self.t), 5), device=device); self.m5[:, 4] = 1

        self.net_sirvd = self.Net_sirvd()
        self.params = list(self.net_sirvd.parameters())
        self.params.extend(list([self.alpha_tilda, self.beta_tilda, \
                                 self.gamma_tilda, self.delta_tilda, self.sigma_tilda]))

    #force parameters to be in a range [0,1]
    def alpha(self):
        return torch.sigmoid(self.alpha_tilda) 
    def beta(self):
        return torch.sigmoid(self.beta_tilda) 
    def gamma(self):
        return torch.sigmoid(self.gamma_tilda)
    def delta(self):
        return torch.sigmoid(self.delta_tilda)
    def sigma(self):
        return torch.sigmoid(self.sigma_tilda)

    class Net_sirvd(nn.Module): 
        def __init__(self):
            super(PINN.Net_sirvd, self).__init__()

            self.fc1=nn.Linear(1, 20) #input t
            self.fc2=nn.Linear(20, 20)
            self.fc3=nn.Linear(20, 20)
            self.fc4=nn.Linear(20, 20)
            self.fc5=nn.Linear(20, 20)
            self.fc6=nn.Linear(20, 20)
            self.fc7=nn.Linear(20, 20)
            self.fc8=nn.Linear(20, 20)
            self.out=nn.Linear(20, 5) #outputs S, I, R, V, D

        def forward(self, t_batch):
            sirvd=F.relu(self.fc1(t_batch))
            sirvd=F.relu(self.fc2(sirvd))
            sirvd=F.relu(self.fc3(sirvd))
            sirvd=F.relu(self.fc4(sirvd))
            sirvd=F.relu(self.fc5(sirvd))
            sirvd=F.relu(self.fc6(sirvd))
            sirvd=F.relu(self.fc7(sirvd))
            sirvd=F.relu(self.fc8(sirvd))
            sirvd=self.out(sirvd)
            return sirvd
    
    def net_f(self, t_batch):
            # pass the timesteps batch to the neural network
            sirvd_hat = self.net_sirvd(t_batch)
            #organize S,I,R,V,D from the neural network's output -- these are normalized values ("hat")
            S_hat, I_hat, R_hat, V_hat, D_hat = sirvd_hat[:,0], sirvd_hat[:,1], sirvd_hat[:,2], sirvd_hat[:,3], sirvd_hat[:,4]

            #S_t
            sirvd_hat.backward(self.m1, retain_graph=True)
            S_hat_t = self.t.grad.clone().to(device)
            self.t.grad.zero_()
            #I_t
            sirvd_hat.backward(self.m2, retain_graph=True)
            I_hat_t = self.t.grad.clone().to(device)
            self.t.grad.zero_()
            #R_t
            sirvd_hat.backward(self.m3, retain_graph=True)
            R_hat_t = self.t.grad.clone().to(device)
            self.t.grad.zero_()
            #V_t
            sirvd_hat.backward(self.m4, retain_graph=True)
            V_hat_t = self.t.grad.clone().to(device)
            self.t.grad.zero_()
            #D_t
            sirvd_hat.backward(self.m5, retain_graph=True)
            D_hat_t = self.t.grad.clone().to(device)
            self.t.grad.zero_()

            # Unnormalize
            S = self.S_min + (self.S_max - self.S_min) * S_hat
            I = self.I_min + (self.I_max - self.I_min) * I_hat
            R = self.R_min + (self.R_max - self.R_min) * R_hat
            V = self.V_min + (self.V_max - self.V_min) * V_hat
            D = self.D_min + (self.D_max - self.D_min) * D_hat

            # Equations
            f1_hat = S_hat_t - ((-(self.beta() / self.N) * S * I) + self.sigma() * R - self.alpha() * S)  / (self.S_max - self.S_min)
            f2_hat = I_hat_t - ((self.beta() / self.N) * S * I - self.gamma() * I - self.delta() * I ) / (self.I_max - self.I_min)
            f3_hat = R_hat_t - (self.gamma() * I - self.sigma() * R) / (self.R_max - self.R_min)
            f4_hat = V_hat_t - (self.alpha() * S) / (self.V_max - self.V_min) 
            f5_hat = D_hat_t - (self.delta() * I) / (self.D_max - self.D_min)

            return f1_hat, f2_hat, f3_hat, f4_hat, f5_hat, S_hat, I_hat, R_hat, V_hat, D_hat
    
    def train(self, n_epochs):
        # train
        print('\nstarting training...\n')

        for epoch in range(n_epochs):
            # lists to hold the output (maintain only the final epoch)
            S_pred_list = []
            I_pred_list = []
            R_pred_list = []
            V_pred_list = []
            D_pred_list = []

            # we pass the timesteps batch into net_f
            f1, f2, f3, f4, f5, S_pred, I_pred, R_pred, V_pred, D_pred = self.net_f(self.t_batch)
            
            self.optimizer.zero_grad() #zero grad
            
            #append the values to plot later (note that we unnormalize them here for plotting)
            S_pred_list.append(self.S_min + (self.S_max - self.S_min) * S_pred)
            I_pred_list.append(self.I_min + (self.I_max - self.I_min) * I_pred)
            R_pred_list.append(self.R_min + (self.R_max - self.R_min) * R_pred)
            V_pred_list.append(self.V_min + (self.V_max - self.V_min) * V_pred)
            D_pred_list.append(self.D_min + (self.D_max - self.D_min) * D_pred)

            #calculate the loss --- MSE of the neural networks output and each compartment
            loss = (torch.mean(torch.square(self.S_hat - S_pred))+ 
                    torch.mean(torch.square(self.I_hat - I_pred))+
                    torch.mean(torch.square(self.R_hat - R_pred))+
                    torch.mean(torch.square(self.V_hat - V_pred))+
                    torch.mean(torch.square(self.D_hat - D_pred))+
                    torch.mean(torch.square(f1))+
                    torch.mean(torch.square(f2))+
                    torch.mean(torch.square(f3))+
                    torch.mean(torch.square(f4))+
                    torch.mean(torch.square(f5))
                    ) 

            loss.backward()
            self.optimizer.step()
            self.scheduler.step() 

            # append the loss value 
            self.losses.append(loss.item())

            if epoch % 1000 == 0:          
                print('\nEpoch ', epoch)
                print('#################################')                

        return S_pred_list, I_pred_list, R_pred_list, V_pred_list, D_pred_list