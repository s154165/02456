

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import add, nn, tensor

class DilatedNet(nn.Module):
    def __init__(self, n_steps_past=16, num_inputs=4,
                                     dilations=[1,1,1],
                                     h1=16, #=16
                                     h2=32, #=10
                                     max_pool_kernel_size = 2,
                                     max_pool_stride = 1,
                                     conv_pad = 1): #[32,32,32,64,64]):

        """
        :param num_inputs: int, number of input variables
        :param h1: int, size of first three hidden layers
        :param h2: int, size of last two hidden layers
        :param dilations: int, dilation value
        :param hidden_units:
        """




        super(DilatedNet, self).__init__()
        self.file_name = os.path.basename(__file__)
        self.hidden_units = [h1,h2]
        self.dilations = dilations

        self.num_inputs = num_inputs

        self.input_width = n_steps_past # n steps past

        self.lstm_dim = self.hidden_units[1]


        ## LSTM
        self.lstm = nn.LSTM(input_size=4,
                         hidden_size=self.hidden_units[0],
                         num_layers=self.lstm_dim,
                         bidirectional=True)

        ## Linear part of the network
        self.lin1 = nn.Linear(in_features= (self.hidden_units[0]*2)*n_steps_past,
                    out_features= 1, #int((self.hidden_units[0]*2)*n_steps_past/2),
                    bias=False)

        # self.lin2 = nn.Linear(in_features= int((self.hidden_units[0]*2)*n_steps_past/2),
        #     out_features=1,
        #     bias=False)

        self.relu = nn.ReLU()

        # Drop out
        # self.dropout = nn.Dropout(0.3)


    def forward(self, x):
        """
        :param x: Pytorch Variable, batch_size x n_stocks x T
        :return:
        """
        x = x.permute(2,0,1)

        x, (h, c) = self.lstm(x)

        x=x.permute(1,0,2)

        x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]) )

        out = self.lin1(x)
        #out = self.relu(x)
        # x = self.dropout(x)

        # out = self.lin2(x)
        # out = self.relu(x)



        return out

