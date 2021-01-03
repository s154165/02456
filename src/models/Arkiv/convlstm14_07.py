
    
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import add, nn, tensor

def compute_conv_dim(dim_input,kernel_size, padding, stride, dilation = 1):
    # return int((dim_input - kernel_size + 2 * padding) / stride + 1)
    return int(((dim_input - dilation*(kernel_size-1) + 2 * padding -1) / stride+1))

def compute_maxpool_dim(dim_input,kernel_size, padding, stride, delation=1):
    return int(((dim_input - delation*(kernel_size-1) + 2 * padding -1) / stride+1))



class DilatedNet(nn.Module):
    def __init__(self, n_steps_past=16, num_inputs=4,
                                     dilations=[1,1,1],
                                     h1=16, #=16
                                     h2=32,
                                     h3 = 10,
                                     h4 = 10,
                                     h5 = 10,#=10
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
        self.lstm_dim=4
        self.hidden_units = [h1,h1,h2,h2,h3,h4,h5]
        self.dilations = dilations
        self.n_steps_past=n_steps_past
        self.num_inputs = num_inputs
        self.receptive_field = sum(dilations) + 1
    
        self.input_width = n_steps_past # n steps past
    
        self.relu = nn.ReLU()
        
        self.length1 = compute_maxpool_dim(compute_conv_dim(self.n_steps_past ,2, 1, 1, dilation = 1), 2, 0, 1)
        self.length2 = compute_maxpool_dim(compute_conv_dim(self.length1 ,2, 1, 1, dilation = 1), 2, 0, 1)
        self.length3 = compute_maxpool_dim(compute_conv_dim(self.length2 ,2, 1, 1, dilation = 1), 2, 0, 1)
        self.length4 = compute_maxpool_dim(compute_conv_dim(self.length3 ,2, 1, 1, dilation = 1), 2, 0, 1)
        self.length5 = compute_maxpool_dim(compute_conv_dim(self.length4 ,2, 1, 1, dilation = 1), 2, 0, 1)
    
        self.max_pool = nn.MaxPool1d(max_pool_kernel_size, stride = max_pool_stride)
        
        #self.dim_conv1 = compute_maxpool_dim(self.hidden_units[0], 2, 1, 1)
        #self.dim_conv2 = compute_maxpool_dim(self.length2, 2, 1,1)
        #self.dim_conv3 = compute_maxpool_dim(compute_conv_dim(self.length3 ,2, 1, 1,self.dilations[2]),max_pool_kernel_size, 0, max_pool_stride,delation=1)

       
        self.conv1 = nn.Conv1d(self.num_inputs, self.hidden_units[0], kernel_size=2, dilation=self.dilations[0],padding=1)
        self.conv2 = nn.Conv1d(self.hidden_units[0], self.hidden_units[1], kernel_size=2, dilation=self.dilations[0],padding=1)
        self.conv3 = nn.Conv1d(self.hidden_units[1], self.hidden_units[2], kernel_size=2, dilation=self.dilations[0],padding=1)
        self.conv4 = nn.Conv1d(self.hidden_units[2], self.hidden_units[3], kernel_size=2, dilation=self.dilations[0],padding=1)
        self.conv5 = nn.Conv1d(self.hidden_units[3], self.hidden_units[4], kernel_size=2, dilation=self.dilations[0],padding=1)
        

        ## LSTM
        self.lstm = nn.LSTM(input_size=self.hidden_units[4],
                         hidden_size=self.hidden_units[5],
                         num_layers=self.lstm_dim,
                         bidirectional=True)        
        
        ## Linear part of the network
        self.lin1 = nn.Linear(in_features=(self.hidden_units[5]*2*self.length5), # self.lstm_dim*(self.hidden_units[3]*2  #skal være self.hidden_units[3] da det er output fra self.lstm_CHO_insulin
                    out_features=self.hidden_units[6],
                    bias=False)
        
        self.lin2 = nn.Linear(in_features= self.hidden_units[6], # self.lstm_dim*(self.hidden_units[3]*2  #skal være self.hidden_units[3] da det er output fra self.lstm_CHO_insulin
                    out_features=1,
                    bias=False)

        # Drop out 
        self.dropout = nn.Dropout(0.3)


       
    def forward(self, x):
        """
        :param x: Pytorch Variable, batch_size x n_stocks x T
        :return:
        """

        current_width = x.shape[2]
        pad = max(self.receptive_field - current_width, 0)
        input_pad = nn.functional.pad(x, [pad, 0], "constant", 0)

        x = self.relu(self.conv1(input_pad))
        x = self.max_pool(x)
       
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
       
        x = self.relu(self.conv3(x))
        x = self.max_pool(x)
        
        x = self.relu(self.conv4(x))
        x = self.max_pool(x)
        
        x = self.relu(self.conv5(x))
        x = self.max_pool(x)

        #print(x.shape)
        x_conv = x.permute(2,0,1)
        
        #print(x.shape)

        x, (h, c) = self.lstm(x_conv)
        
   
        x=x.permute(1,0,2)
       
        x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]) )


        features_final = self.relu(x)
        features_final = self.lin1(features_final)
        
        out = self.lin2(features_final)

        
        return out