import torch
import torch.nn as nn

import numpy as np

# C51
Vmax = 0.02
Vmin = -0.02
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, N_ATOMS)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions * N_ATOMS)
        )

        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val_out = self.fc_val(conv_out).view(x.size()[0], 1, N_ATOMS)
        adv_out = self.fc_adv(conv_out).view(x.size()[0], -1, N_ATOMS)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + (adv_out - adv_mean)

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())
        



class LSTM_Forex (nn.Module):
    def __init__(self,selDevice,input_shape,actions):

        super(LSTM_Forex,self).__init__()
        self.input_shape = input_shape
        self.actions = actions
        self.selected_device = selDevice
        self.inSize = self.input_shape[1]
        self.hiddenSize = 200
        self.numLayers = 2
        self.outSize = 512
        self.lstm = nn.LSTM(self.inSize,self.hiddenSize,self.numLayers,batch_first=True)
        
        


        self.fc_val = nn.Sequential(
            nn.Linear(self.hiddenSize, 512),
            nn.ReLU(),
            nn.Linear(512, N_ATOMS)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(self.hiddenSize, 512),
            nn.ReLU(),
            nn.Linear(512, self.actions * N_ATOMS)
        )

        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        h0 = torch.zeros(self.numLayers,x.size(0),self.hiddenSize,device=self.selected_device)
        c0 = torch.zeros(self.numLayers,x.size(0),self.hiddenSize,device=self.selected_device)
        out,(hn,cn) = self.lstm(x,(h0,c0))
        val_out = self.fc_val(out[:,-1,:]).view(x.size()[0], 1, N_ATOMS)
        adv_out = self.fc_adv(out[:,-1,:]).view(x.size()[0], -1, N_ATOMS)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + (adv_out - adv_mean)
    

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())


    
    

