import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F





class Residual(nn.Module):
    def __init__(self,in_channel,R=8,k=2):
        super(Residual, self).__init__()
        self.avg=nn.AdaptiveAvgPool2d((1,1))
        self.relu=nn.ReLU(inplace=True)
        self.R=R
        self.k=k
        out_channel=int(in_channel/R)
        self.fc1=nn.Linear(in_channel,out_channel)
        fc_list=[]
        for i in range(k):
          fc_list.append(nn.Linear(out_channel,2*in_channel))
        self.fc2=nn.ModuleList(fc_list)


    def forward(self,x):
        x=self.avg(x)
        x=torch.squeeze(x)
        x=self.fc1(x)
        x=self.relu(x)
        result_list=[]
        for i in range(self.k):
            result=self.fc2[i](x)
            result=2*torch.sigmoid(result)-1
            result_list.append(result)
        return result_list

    

      

    





class Dynamic_relu_b(nn.Module):
    def __init__(self,inchannel,R=8,k=2):
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

       
        self.residual=Residual(inchannel)

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


class Dynamic_Relu_A(nn.Module):
    def __init__(self, R=8, k=2):
        super(Dynamic_Relu_A, self).__init__()
        self.lambda_alpha = 1
        self.lambda_beta = 0.5
        self.R = R
        self.k = k
        self.init_alpha = torch.zeros(self.k)
        self.init_beta = torch.zeros(self.k)
        self.init_alpha[0] = 1
        self.init_beta[0] = 1

        for i in range(1, k):
            self.init_alpha[i] = 0
            self.init_beta[i] = 0

        self.residual = residual('a',self.R, self.k)

    def forward(self,x):
        '''
        :param input: [N,C,H,W]
        :return: [N,C,H,W]
        '''
        delta = self.residual(input)
        bs = input.shape[0]
        alpha = torch.zeros((self.k, bs))
        beta = torch.zeros((self.k, bs))
        for i in range(self.k):
            alpha[i,:]=delta[i][:,0]
            beta[i,:]=delta[i][:,1]
        alpha1 = alpha[0]
        beta1 = beta[0]
        max_result = self.dynamic_function(alpha1, beta1, input, 0)
        for i in range(1, self.k):
                alphai = alpha[i]
                betai = beta[i]
                result = self.dynamic_function(alphai, betai, input, i)
                max_result = torch.max(max_result, result)
        return max_result


    def dynamic_function(self,alpha,beta,x,k):
        init_alpha=self.init_alpha[k]
        init_beta=self.init_beta[k]
        alpha=init_alpha+self.lambda_alpha*alpha
        beta=init_beta+self.lambda_beta*beta
        bs=x.shape[0]
        results=torch.zeros_like(x)
        for i  in range(bs):
          results[i]=alpha[i]*x[i]+beta[i]
        return results




'''relu_c normalize'''
def normalize(z,gamma,t):
    bs=z.shape[0]
    results=torch.zeros_like(z)
    for i in range(bs):
        x=z[i]
        x=x/t
        x_exp=torch.exp(x)
        x_sum=torch.sum(x_exp)
        '''softmax'''
        result=gamma*x_exp/x_sum
        '''min(result,1)'''
        result[result>1]=1
        results[i]=result
    return results


class Spatial_attentions(nn.Module):
    def __init__(self,t=10,divisor=3):
        super(Spatial_attentions, self).__init__()
        self.t=t
        self.divisor=divisor

    def forward(self,x):
        '''

        :param x: [N,C,H,W]
        :return:[N,1,H,W] ,and all elements in this tensor are less than 1
        '''
        H,W=x.shape[2:]
        gamma=H*W/self.divisor
        in_channel=x.shape[1]
        x=nn.Conv2d(in_channel,1,kernel_size=1)(x)
        x=normalize(x,gamma,self.t)
        return x






class Dynamic_Relu_C(nn.Module):
    def __init__(self, t=10,divisor=3,R=8, k=2):
        super(Dynamic_Relu_C, self).__init__()
        self.lambda_alpha = 1
        self.lambda_beta = 0.5
        self.R = R
        self.k = k
        self.init_alpha = torch.zeros(self.k)
        self.init_beta = torch.zeros(self.k)
        self.init_alpha[0] = 1
        self.init_beta[0] = 1

        for i in range(1, k):
            self.init_alpha[i] = 0
            self.init_beta[i] = 0

        self.residual = residual('c', self.R, self.k)
        self.spatial=Spatial_attentions(t,divisor)


    def forward(self,input):
        '''

        :param input: [N,C,H,W]
        :return: [N,C,H,W]
        '''
        delta=self.residual(input)
        spatial=self.spatial(input)
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
        max_result=self.dynamic_function(alpha1,beta1,input,0,spatial)
        for i in range(1,self.k):
            alphai=alpha[i]
            betai=beta[i]
            result=self.dynamic_function(alphai,betai,input,i,spatial)
            max_result=torch.max(max_result,result)
        return max_result



    def dynamic_function(self,alpha,beta,x,k,spatial):
        init_alpha=self.init_alpha[k]
        init_beta=self.init_beta[k]
        alpha=init_alpha+self.lambda_alpha*alpha
        beta=init_beta+self.lambda_beta*beta
        bs=x.shape[0]
        channel=x.shape[1]
        results=torch.zeros_like(x)
        for i  in range(bs):
            pai=spatial[i]
            for c in range(channel):
                '''alpha and beta will become [1,H,W] which will apply to each channel '''
                results[i,c,:,:]=x[i,c]*alpha[i,c]*pai+beta[i,c]*pai
        return results























