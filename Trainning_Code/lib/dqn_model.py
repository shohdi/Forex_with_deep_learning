import torch
import torch.nn as nn

import numpy as np


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
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)



class LSTM_Forex (nn.Module):
    def __init__(self,selDevice,input_shape,actions):

        super(LSTM_Forex,self).__init__()
        self.input_shape = input_shape
        self.actions = actions
        self.selected_device = selDevice
        self.inSize = self.input_shape[1]
        self.hiddenSize = 100
        self.numLayers = 2
        self.outSize = 512
        self.lstm = nn.LSTM(self.inSize,self.hiddenSize,self.numLayers,batch_first=True)
        
        
        self.fc = nn.Sequential(
            nn.Linear(self.hiddenSize, 512),
            nn.ReLU(),
            nn.Linear(512, self.actions)
        )
    
    def forward(self,x):
        h0 = torch.zeros(self.numLayers,x.size(0),self.hiddenSize,device=self.selected_device)
        c0 = torch.zeros(self.numLayers,x.size(0),self.hiddenSize,device=self.selected_device)
        out,(hn,cn) = self.lstm(x,(h0,c0))
        out = self.fc(out[:,-1,:])
        return out

    
    

