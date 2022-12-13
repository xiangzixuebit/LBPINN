#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import argparse
import sys
sys.path.append("..")
import math
import torch
from torch.autograd import grad 
import numpy as np


def gradients(u, x):
    return grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True,  only_inputs=True, allow_unused=True)[0]


def loss_u(u_p, t_u):
    return ((u_p - t_u) ** 2).mean()

def Sirenloss_f(u_f, u_f_coords, args):
    u_x = gradients(u_f, u_f_coords)[:, 0:1]
    u_t = gradients(u_f, u_f_coords)[:, 1:2]
    u_xx = gradients(u_x, u_f_coords)[:, 0:1]
    f = u_t + u_f * u_x - args.nu * u_xx
    return f, (f**2).mean()

def loss_f(u_f, x_f, t_f, args):
    u_x = gradients(u_f, x_f)
    u_t = gradients(u_f, t_f)
    u_xx = gradients(u_x, x_f)
    f = u_t + u_f * u_x - args.nu * u_xx
    return f, (f**2).mean()

def loss_b(u_1,u_2, u_3, x):
    return ((u_1 + torch.sin(math.pi * x)) ** 2).mean() + (u_2 ** 2).mean() + (u_3 ** 2).mean()



