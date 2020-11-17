#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import torch 


# In[2]:


class DefaultConfig(object):
    
    env = 'beikeQA'  
    vis_port =8097 
    
    MAX_SEQUENCE_LENGTH = 64
    input_categories = ['q1','q2']
    output_categories = 'label'
    device = 'cuda'
    num_workers = 0
    
    lr = 2e-5
    weight_decay = 1e-5
    adam_epsilon = 1e-8
    n_epochs = 10
    early_stop = 10
    
    train_size = 32
    pred_size = 512
    

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()
