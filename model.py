#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from itertools import repeat


"""
不同于常见的dropout层，
传统的dropout是元素级别的，对矩阵内元素随机置0；
Spatial_Dropout层则是字/词向量级别:如"你好哇李银河",对"哇"字向量的所有元素置0；
适用于NLP任务。
用于embedding层后。
"""
class Spatial_Dropout(nn.Module):
    def __init__(self,drop_prob):

        super(Spatial_Dropout,self).__init__()
        self.drop_prob = drop_prob

    def forward(self,inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output

    def _make_noise(self,inputs):
        return inputs.new().resize_(inputs.size(0),*repeat(1, inputs.dim() - 2),inputs.size(2))
    

class Net(nn.Module):
    """
    B为batch_size,S为seq_length;
    input:(q_id,q_mask,q_atn),即bert格式的输入;
    q_id/q_mask/q_atn.shape=[B,S];
    
    output.shape=[B,1]；即输出0~1间的数值;(概率值)；
    """
    def __init__(self,PreModel):
        super().__init__()
        self.PreModel = PreModel
        self.features = self.PreModel.config.hidden_size
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1,self.features))
        self.max_pool = nn.AdaptiveMaxPool2d((1,self.features))
        self.sp_drop_out = Spatial_Dropout(0.2)
        self.drop_out = nn.Dropout()
        self.fc = nn.Linear(self.features*5,1)
        self.active = nn.Sigmoid()
        
    def forward(self,q_id,q_mask,q_atn):
        """
        H=hidden_size=768
        
        outputs包含两个输出,第一个为hidden_layer,输出为embedding，embedding.shape=[B,S,H]官方建议在文本分类任务中通过max_pool等手段提取有效信息
        第二个输出pooler_output,pooler_output.shape=[B,H],可直接接入Dense层进行分类任务；
        """
        if self.training:
            self.PreModel.train()
            outputs = self.PreModel(q_id,attention_mask=q_mask,token_type_ids=q_atn)
        else:
            self.PreModel.eval()
            with torch.no_grad():
                outputs = self.PreModel(q_id,attention_mask=q_mask,token_type_ids=q_atn)
                
                
        q_embedding = outputs[0]
        q_embedding = self.sp_drop_out(q_embedding)#在embedding层后接Spatial_Dropout层能稳定提高模型的泛化能力
        output = outputs[1]
        
        q = self.avg_pool(q_embedding).squeeze(1)
        a = self.max_pool(q_embedding).squeeze(1)
        t = q_embedding[:,-1]
        e = q_embedding[:,0]
        x = torch.cat([q,a,t,e,output],dim=-1)
        
        x = self.fc(self.drop_out(x))
        
        x = self.active(x)
        return x
