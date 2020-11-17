#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from transformers import BertTokenizer,BertModel
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from config import opt
from model import Net
from utils import dataset
from transformers import BertConfig
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
from sklearn.metrics import roc_auc_score,f1_score
import copy
import os
from visualize import Visualizer
from utils import AdamW,WarmupLinearSchedule


# In[15]:


def _convert_to_transformer_inputs(question, answer, tokenizer, max_sequence_length):
    """
    question="你好吗？";answer="我很好。"，
    输出为[input_ids_q, input_masks_q, input_segments_q],即Bert格式的输入；
    """
    def return_id(str1, str2, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1, str2,
            add_special_tokens=True,
            max_length=length,
            truncation_strategy=truncation_strategy,
            #truncation=True
            )
        
        input_ids =  inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        
        return [input_ids, input_masks, input_segments]
    
    input_ids_q, input_masks_q, input_segments_q = return_id(question, answer, 'longest_first', max_sequence_length)
    
    return [input_ids_q, input_masks_q, input_segments_q]


# In[3]:


def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    """
    生成三个输入矩阵：input_ids_q，input_masks_q，input_segments_q
    """
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        q, a = instance.q1, instance.q2

        ids_q, masks_q, segments_q= _convert_to_transformer_inputs(q, a, tokenizer, max_sequence_length)
        
        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

    return [np.asarray(input_ids_q, dtype=np.int32), 
            np.asarray(input_masks_q, dtype=np.int32), 
            np.asarray(input_segments_q, dtype=np.int32)]


# In[4]:


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


# In[41]:


def fit(idx,model,train_iter,valid_iter,optimizer,scheduler,criterion,epochs,vis):
    best_model_dict = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    
    vis.log("First eval on valid-data:")
    auc = Eval(model,valid_iter)
    vis.plot('{idx} valid-AUC'.format(idx=idx),auc)
    
    best_auc = auc
    no_gain_rc = 0
    
    #每轮训练,作出20个点(绘制损失曲线)，评估5次模型
    plot_loss_time = len(train_iter)//20
    eval_time = len(train_iter)//5
    
    break_flag = 0
    
    for epoch in range(1,epochs+1):
        vis.log("=========TRAIN and EVAL at epoch={epoch}=========".format(epoch = epoch))
        
        model.train()
        loss_all,outputs_all,preds_all = [],[],[]
        
        for i,batch in enumerate(tqdm(train_iter)):
            input_ids_q,input_masks_q,input_segments_q,outputs = batch
            optimizer.zero_grad()
            y_pred = model.forward(input_ids_q,input_masks_q,input_segments_q)
            
            loss = criterion(y_pred,outputs)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            loss_all.append(loss.to('cpu').item())
            outputs_all += outputs.reshape(-1).cpu().detach().numpy().tolist()
            preds_all += y_pred.reshape(-1).cpu().detach().numpy().tolist()
            
            if i%plot_loss_time == 0 and i > 0:
                vis.plot('{idx} train-BCELoss'.format(idx=idx),np.mean(loss_all))
                vis.plot('{idx} train-AUC'.format(idx=idx),roc_auc_score(outputs_all,preds_all))
                loss_all,outputs_all,preds_all = [],[],[]
            
            if i%eval_time == 0:
                vis.log("Eval on Devset...")
                auc = Eval(model,valid_iter)
                vis.plot('{idx} valid-AUC'.format(idx=idx),auc)
                model.train()
                if auc > best_auc:
                    if not ((i==0) and (epoch==1)):
                        vis.log("valid-AUC 由{:.3f} 更新至 {:.3f}".format(best_auc,auc))
                        best_auc = auc
                        best_model_dict = copy.deepcopy(model.state_dict())#保存最优参数
                        no_gain_rc = 0
                else:
                    no_gain_rc += 1
                    if no_gain_rc >= opt.early_stop:
                        vis.log("连续{}个epoch没有提升，在epoch={}提前终止".format(no_gain_rc,epoch))
                        break_flag = 1
                        break
        if break_flag:
            break
        
    model.load_state_dict(best_model_dict)#使用最优参数
    return model

# In[38]:


def Eval(model,iterator):
    model.eval()
    outputs_all,preds_all = [],[]
    
    for i,batch in enumerate(iterator):
        input_ids_q,input_masks_q,input_segments_q,outputs = batch
        y_pred = model.forward(input_ids_q,input_masks_q,input_segments_q)
        outputs_all += outputs.reshape(-1).cpu().detach().numpy().tolist()
        preds_all += y_pred.reshape(-1).cpu().detach().numpy().tolist()
    
    auc = roc_auc_score(outputs_all,preds_all)#AUC作为评估标准
    return auc


# In[32]:


def predict(model,iterator):
    model.eval()
    preds_all = []
    
    for i,batch in enumerate(iterator):
        input_ids_q,input_masks_q,input_segments_q = batch
        y_pred = model.forward(input_ids_q,input_masks_q,input_segments_q)
        preds_all += y_pred.reshape(-1).cpu().detach().numpy().tolist()
        
    return np.array(preds_all).reshape(-1,1)


# In[31]:


def search_f1(y_true, y_pred):
    best = 0
    best_t = 0
    for i in range(30,60):
        tres = i / 100
        y_pred_bin =  (y_pred > tres).astype(int)
        score = f1_score(y_true, y_pred_bin)
        if score > best:
            best = score
            best_t = tres
    return best, best_t


# In[42]:


def main(**kwargs):
    opt._parse(kwargs)
    vis = Visualizer(opt.env,port=opt.vis_port)
    
    df_train = pd.read_csv('data/df_train.csv')
    df_test = pd.read_csv('data/df_test.csv')
    
#     tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
    
    inputs = compute_input_arrays(df_train,opt.input_categories,tokenizer,opt.MAX_SEQUENCE_LENGTH)
    test_inputs = compute_input_arrays(df_test,opt.input_categories,tokenizer,opt.MAX_SEQUENCE_LENGTH)
    outputs = compute_output_arrays(df_train,opt.output_categories)
    
    valid_preds = []
    test_preds = []
    oof = np.zeros((len(df_train),1))
    
    gkf = GroupKFold(n_splits=10).split(X=df_train.q2, groups=df_train.id)
    
    test_dataset = dataset(test_inputs,opt.device)
    test_iter = data.DataLoader(dataset=test_dataset,
                                   batch_size=opt.pred_size,shuffle=False,
                                   num_workers=opt.num_workers)
    
    for fold, (train_idx, valid_idx) in enumerate(gkf):
        train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
        train_outputs = outputs[train_idx]
        valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
        valid_outputs = outputs[valid_idx]
        
        train_dataset = dataset(train_inputs,opt.device,train_outputs)
        valid_dataset = dataset(valid_inputs,opt.device,valid_outputs)
        oof_dataset = dataset(valid_inputs,opt.device)
        
        """
        oof_iter和dev_iter都是验证集数据；但是dev_iter用于在训练模型的过程中评估auc分数，避免过拟合；
        oof_iter用于在模型训练完后对验证集数据做出预测，该预测值将用于评估模型的线下的f1分数;
        """
        train_iter = data.DataLoader(dataset=train_dataset,
                             batch_size=opt.train_size,shuffle=True,
                             num_workers=opt.num_workers)
        
        dev_iter = data.DataLoader(dataset=valid_dataset,
                            batch_size=opt.pred_size,shuffle=True,
                            num_workers=opt.num_workers)
        
        oof_iter = data.DataLoader(dataset=oof_dataset,
                            batch_size=opt.pred_size,shuffle=False,
                            num_workers=opt.num_workers)
        
        if os.path.exists(opt.model_path):
            vis.log("=========载入模型===========")
            model = torch.load(opt.model_path)
            
        else:
            vis.log("=========初始模型===========")
#             PreModel = BertModel.from_pretrained('bert-base-chinese')
            PreModel = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
            model = Net(PreModel)
            
        model.to(opt.device)
        if opt.num_workers !=0:
            model = nn.DataParallel(model,device_ids=[0,1])
    
        optimizer = AdamW(model.parameters(),lr=opt.lr,weight_decay=opt.weight_decay,eps=opt.adam_epsilon)
        
        total_steps = len(train_iter)*opt.n_epochs
        warmup_steps = total_steps//10
        scheduler = WarmupLinearSchedule(optimizer,warmup_steps=warmup_steps,t_total=total_steps)
        
        criterion = nn.BCELoss()
        
        vis.log("=========Model Fit at Fold = {fold}=========".format(fold=fold))
        model = fit(fold,model,train_iter,dev_iter,optimizer,scheduler,criterion,opt.n_epochs,vis)
        oof_p = predict(model,oof_iter)
        oof[valid_idx] = oof_p
        test_preds.append(predict(model,test_iter))
        f1,t = search_f1(valid_outputs,oof_p.reshape(-1))
        vis.log("!=======================================!")
        vis.log('fold = {} validation score = {:.3f}'.format(fold,f1))
        vis.log('fold = {} thres = {:.3f}'.format(fold,t))
        vis.log("!=======================================!")
    
    best_score,best_t = search_f1(outputs,oof.reshape(-1))
    vis.log("!!!=======================================!!!")
    vis.log('best score = {:.3f}'.format(best_score))
    vis.log('thres = {:.3f}'.format(best_t))
    vis.log("!!!=======================================!!!")
    
    sub = np.average(test_preds, axis=0)
    sub = sub > best_t
    df_test['label'] = sub.astype(int)
    df_test[['id','id_sub','label']].to_csv('submission_beike_{}.csv'.format(best_score),index=False, header=None,sep='\t')
    vis.log('Finish')

if __name__=='__main__':
    import fire
    fire.Fire()

