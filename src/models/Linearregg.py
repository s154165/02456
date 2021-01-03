#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:21:59 2020

@author: annpham
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import add, nn, tensor



class DilatedNet(nn.Module):
    def __init__(self, n_steps_past=16, num_inputs=4,
                                     h1=0, #=16
                                     h2=0): #[32,32,32,64,64]):
        
        """
        :param num_inputs: int, number of input variables
        :param h1: int, size of first three hidden layers
        :param h2: int, size of last two hidden layers
        :param dilations: int, dilation value
        :param hidden_units:
        """
        

     
        
        super(DilatedNet, self).__init__()
        self.file_name = os.path.basename(__file__)
        
        ## Linear part of the network
        self.n_steps_past = n_steps_past
        self.num_inputs=num_inputs
        
        self.lin1 = nn.Linear(in_features=self.n_steps_past*self.num_inputs, #skal være self.hidden_units[3] da det er output fra self.lstm_CHO_insulin
                    out_features=1,
                    bias=False)


       
    def forward(self, x):
        
        """
        
        :param x: Pytorch Variable, batch_size x n_stocks x T
        :return:
        """
 
        x=torch.reshape(x, (x.shape[0], self.n_steps_past*self.num_inputs))


        out_final = self.lin1(x)
        
        return out_final
