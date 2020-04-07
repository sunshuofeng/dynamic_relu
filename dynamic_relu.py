import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F




def make_residual_fc(input,R=8,k=2):
    in_channel=input.shape[1]
    out_channel=int(in_channel/R)
    fc1=nn.Linear(in_channel,out_channel)
    fc_list=[]
    for i in range(k):
        fc_list.append(nn.Linear(out_channel,2*in_channel))
    fc2=nn.ModuleList(fc_list)
    return fc1,fc2


class residual(nn.Module):
    def __init__(self,R=8,k=2):
        super(residual, self).__init__()
        self.avg=nn.AdaptiveAvgPool2d((1,1))
        self.relu=nn.ReLU(inplace=True)
        self.R=R
        self.k=k

    def forward(self,x):
        x=self.avg(x)

        fc1,fc2=make_residual_fc(x,self.R,self.k)
        x=torch.squeeze(x)
        x=fc1(x)
        x=self.relu(x)
        result_list=[]
        for i in range(self.k):
            result=fc2[i](x)
            result=2*torch.sigmoid(result)-1
            result_list.append(result)
        return result_list





class Dynamic_relu_b(nn.Module):
    def __init__(self,R=8,k=2):
        super(Dynamic_relu_b, self).__init__()
        self.lambda_alpha=1
        self.lambda_beta=0.5
        self.R=R
        self.k=k
        self.init_alpha=torch.zeros(self.k)
        self.init_beta=torch.zeros(self.k)
        self.init_alpha[0]=1
        self.init_beta[0]=1
        for i in range(1,k):
            self.init_alpha[i]=0
            self.init_beta[i]=0

        self.residual=residual(self.R,self.k)


    def forward(self,input):
        delta=self.residual(input)
        in_channel=input.shape[1]
        bs=input.shape[0]
        alpha=torch.zeros((self.k,bs,in_channel))
        beta=torch.zeros((self.k,bs,in_channel))
        for i in range(self.k):
            for j,c in enumerate(range(0,in_channel*2,2)):
                alpha[i,:,j]=delta[i][:,c]
                beta[i,:,j]=delta[i][:,c+1]
        alpha1=alpha[0]
        beta1=beta[0]
        max_result=self.dynamic_function(alpha1,beta1,input,0)
        for i in range(1,self.k):
            alphai=alpha[i]
            betai=beta[i]
            result=self.dynamic_function(alphai,betai,input,i)
            max_result=torch.max(max_result,result)
        return max_result


    def dynamic_function(self,alpha,beta,x,k):
        init_alpha=self.init_alpha[k]
        init_beta=self.init_beta[k]
        alpha=init_alpha+self.lambda_alpha*alpha
        beta=init_beta+self.lambda_beta*beta
        bs=x.shape[0]
        channel=x.shape[1]
        results=torch.zeros_like(x)
        for i  in range(bs):
            for c in range(channel):
                results[i,c,:,:]=x[i,c]*alpha[i,c]+beta[i,c]
        return results


















